import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class TensorBoardLogger:
    """Encapsulates all logging, checkpoint loading/saving, and TensorBoard management."""
    
    def __init__(self, training_type='rl_training', pretrained_model_path=None, hyperparams=None):
        """
        Initialize the TensorBoardLogger.
        
        Args:
            training_type: 'rl_training' or 'human_feedback'
            pretrained_model_path: Path to checkpoint to resume from (optional)
            hyperparams: Dictionary of hyperparameters to log (optional)
        """
        self.training_type = training_type
        self.pretrained_model_path = pretrained_model_path
        self.hyperparams = hyperparams or {}
        
        # Tracking variables
        self.start_epoch = 0
        self.best_metric = 0  # best_win_rate for RL, best_val_accuracy for human_feedback
        self.checkpoint_data = None
        
        # TensorBoard writer (initialized later)
        self.writer = None
        self.log_dir = None
        self.is_resuming = False
        
    def print_header(self):
        """Print training header with hyperparameters."""
        print("=" * 60)
        print(f"Training Type: {self.training_type}")
        for key, value in self.hyperparams.items():
            print(f"{key}: {value}")
        if self.pretrained_model_path:
            print(f"Loading checkpoint: {self.pretrained_model_path}")
        print("=" * 60)
        
    def load_checkpoint(self, model, optimizer=None, use_best_checkpoint=False):
        """
        Load checkpoint and restore model/optimizer state.
        
        Args:
            model: PyTorch model to load weights into
            optimizer: PyTorch optimizer to restore state (optional)
            use_best_checkpoint: If True, load model_best.pth; otherwise load model_last.pth (default: False)
            
        Returns:
            Tuple of (start_epoch, best_metric)
        """
        if not self.pretrained_model_path:
            print("No checkpoint specified - starting from scratch\n")
            return 0, 0
            
        # Determine checkpoint path
        if os.path.isdir(self.pretrained_model_path):
            # Choose between best and last checkpoint
            checkpoint_filename = 'model_best.pth' if use_best_checkpoint else 'model_last.pth'
            checkpoint_path = os.path.join(self.pretrained_model_path, checkpoint_filename)
        else:
            checkpoint_path = self.pretrained_model_path
            
        if not os.path.exists(checkpoint_path):
            print(f"⚠ Warning: Checkpoint not found: {checkpoint_path}")
            print("  Starting from scratch...\n")
            return 0, 0
            
        try:
            checkpoint = torch.load(checkpoint_path)
            
            # Load model weights
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                
                # Load optimizer state if available and optimizer provided
                if optimizer and 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("  ✓ Optimizer state restored")
                elif optimizer:
                    print("  ⚠ No optimizer state found (loss may jump)")
                
                # Get epoch and best metric info
                self.start_epoch = checkpoint.get('epoch', 0) + 1
                
                # Handle different checkpoint formats
                if 'best_win_rate' in checkpoint:
                    self.best_metric = checkpoint['best_win_rate']
                    metric_name = 'win_rate'
                elif 'best_val_accuracy' in checkpoint:
                    self.best_metric = checkpoint['best_val_accuracy']
                    metric_name = 'val_accuracy'
                else:
                    self.best_metric = 0
                    metric_name = 'metric'
                
                checkpoint_type = "best" if use_best_checkpoint else "last"
                print(f"✓ Checkpoint loaded ({checkpoint_type}) - resuming from epoch {self.start_epoch}")
                if self.best_metric > 0:
                    print(f"  Best {metric_name}: {self.best_metric:.3f}")
                
                # Store checkpoint for later reference
                self.checkpoint_data = checkpoint
                
            else:
                # Old format - just model weights
                model.load_state_dict(checkpoint, strict=False)
                print(f"✓ Checkpoint loaded (old format - starting from epoch 0)")
                self.start_epoch = 0
                self.best_metric = 0
                
        except Exception as e:
            print(f"⚠ Warning: Failed to load checkpoint: {e}")
            print("  Starting from scratch...")
            self.start_epoch = 0
            self.best_metric = 0
            
        print()
        return self.start_epoch, self.best_metric
    
    def setup_tensorboard(self):
        """
        Setup TensorBoard writer, always creating a new directory.
        If resuming from a checkpoint, track the previous run in hyperparameters.
        
        Returns:
            Tuple of (writer, log_dir, is_resuming)
        """
        # Always create new timestamped run directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = f'runs/{self.training_type}/{timestamp}'
        
        # Check if resuming from a checkpoint
        previous_run_id = None
        if self.pretrained_model_path and os.path.exists(self.pretrained_model_path):
            # Extract the previous run ID from checkpoint path
            # Handle both directory paths and file paths
            if os.path.isdir(self.pretrained_model_path):
                # Path is a directory: runs/human_feedback/20260202_181723
                previous_run_id = os.path.basename(self.pretrained_model_path)
            else:
                # Path is a file: runs/human_feedback/20260202_181723/model_last.pth
                checkpoint_dir = os.path.dirname(self.pretrained_model_path)
                previous_run_id = os.path.basename(checkpoint_dir)
            
            self.is_resuming = True
            print(f"Resuming from checkpoint: {previous_run_id}")
            print(f"Creating new run - TensorBoard logging to: {self.log_dir}")
        else:
            self.is_resuming = False
            print(f"Starting new training - TensorBoard logging to: {self.log_dir}")
        
        self.writer = SummaryWriter(self.log_dir)
        
        # Log hyperparameters (numeric values only)
        if self.hyperparams:
            self.writer.add_hparams(self.hyperparams, {})
        
        # Log previous run ID as text summary (TensorBoard add_hparams doesn't support strings)
        if previous_run_id:
            self.writer.add_text('Training/resumed_from', previous_run_id, 0)
        
        print()
        return self.writer, self.log_dir, self.is_resuming
    
    def get_checkpoint_paths(self):
        """
        Get paths for saving best and last checkpoints.
        
        Returns:
            Tuple of (best_checkpoint_path, last_checkpoint_path)
        """
        best_checkpoint_path = os.path.join(self.log_dir, 'model_best.pth')
        last_checkpoint_path = os.path.join(self.log_dir, 'model_last.pth')
        return best_checkpoint_path, last_checkpoint_path
    
    def save_checkpoint(self, epoch, model, optimizer, metric_value, metric_name='metric', 
                       is_best=False, additional_data=None):
        """
        Save checkpoint (best and/or last).
        
        Args:
            epoch: Current epoch number
            model: PyTorch model to save
            optimizer: PyTorch optimizer to save
            metric_value: Current value of the tracking metric
            metric_name: Name of the metric ('win_rate', 'val_accuracy', etc.)
            is_best: Whether this is the best checkpoint so far
            additional_data: Additional data to save in checkpoint (optional dict)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            f'best_{metric_name}': self.best_metric
        }
        
        # Add any additional data
        if additional_data:
            checkpoint.update(additional_data)
        
        best_path, last_path = self.get_checkpoint_paths()
        
        # Always save last checkpoint
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint if this is the best
        if is_best:
            self.best_metric = metric_value
            checkpoint[f'best_{metric_name}'] = self.best_metric
            torch.save(checkpoint, best_path)
    
    def log_scalar(self, tag, value, step):
        """Log a scalar value to TensorBoard."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, scalar_dict, step):
        """Log multiple scalar values to TensorBoard."""
        if self.writer:
            for tag, value in scalar_dict.items():
                self.writer.add_scalar(tag, value, step)
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.writer:
            self.writer.close()
    
    def print_completion_summary(self, final_summary):
        """
        Print training completion summary.
        
        Args:
            final_summary: Dictionary with summary information
        """
        best_path, last_path = self.get_checkpoint_paths()
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        for key, value in final_summary.items():
            print(f"{key}: {value}")
        print(f"Best model saved to: {best_path}")
        print(f"Last model saved to: {last_path}")
        print(f"TensorBoard logs saved to: {self.log_dir}")
        print("=" * 60)