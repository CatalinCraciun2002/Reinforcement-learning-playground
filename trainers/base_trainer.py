"""
Base Trainer Abstract Class

Provides a standardized interface for all RL training scripts with:
- TensorBoard logging
- Checkpoint management (save/load)
- Validation loops
- Progress bar management
- Hyperparameter tracking
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runs.logger import TensorBoardLogger


class BaseTrainer(ABC):
    """
    Abstract base class for all RL trainers.
    
    Provides standardized:
    - Logging (TensorBoard + progress bars)
    - Checkpointing (save/load with metadata)
    - Validation
    - Training loop structure
    
    Subclasses must implement algorithm-specific methods.
    """
    
    @staticmethod
    def build_parser():
        """
        Builds a standard argument parser with common RL hyperparameters.
        Subclasses can call this and then add their own specific arguments.
        """
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs to train')
        parser.add_argument('--batch-size', type=int, default=4, help='Batch size (number of parallel environments)')
        parser.add_argument('--steps-per-epoch', type=int, default=16, help='Steps per epoch per environment')
        parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
        parser.add_argument('--lam', type=float, default=0.9, help='GAE lambda')
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        parser.add_argument('--clip-epsilon', type=float, default=0.2, help='PPO clip epsilon')
        parser.add_argument('--ppo-epochs', type=int, default=4, help='PPO optimization epochs')
        parser.add_argument('--mini-batch-size', type=int, default=4, help='PPO mini batch size')
        parser.add_argument('--show-epochs', type=int, default=5, help='Show validation game every N epochs')
        parser.add_argument('--validation-games', type=int, default=2, help='Number of validation games')
        parser.add_argument('--train-suite', type=str, default='custom_only', help='Training environment suite')
        parser.add_argument('--test-suite', type=str, default='custom_only', help='Testing environment suite')
        parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
        parser.add_argument('--use-best', action='store_true', default=False, help='Use best checkpoint when resuming')
        parser.add_argument('--save-visualization', action='store_true', default=False, help='Save data for visualization')
        return parser

    def __init__(
        self,
        training_type: str,
        args=None,
        num_epochs: int = None,
        hyperparams: Dict[str, Any] = None,
        resume_from: Optional[str] = None,
        use_best_checkpoint: bool = False,
        save_visualization_data: bool = False,
        **kwargs
    ):
        """
        Initialize base trainer.
        
        Args:
            training_type: Type of training (e.g., 'dqn', 'qlearning', 'rl_training', 'human_feedback')
                          This determines the subdirectory in runs/ where logs are saved
            args: Optional argparse.Namespace. If provided, automatically binds hyperparams.
            num_epochs: Number of epochs to train (legacy)
            hyperparams: Dictionary of hyperparameters to log (legacy)
            resume_from: Path to checkpoint to resume from (optional)
            use_best_checkpoint: If True and resume_from is a directory, load model_best.pth
            save_visualization_data: If True, save detailed training data for visualization (default: False)
        """
        self.training_type = training_type
        
        if args is not None:
            # New refactored path
            self.args = args
            self.num_epochs = getattr(args, 'num_epochs', 100)
            self.resume_from = getattr(args, 'resume', None)
            self.use_best_checkpoint = getattr(args, 'use_best', False)
            self.save_visualization_data = getattr(args, 'save_visualization', False)
            
            # Common RL arguments bound to self if they exist in args
            for attr in ['batch_size', 'steps_per_epoch', 'gamma', 'lam', 'lr', 
                         'clip_epsilon', 'ppo_epochs', 'mini_batch_size', 'show_epochs', 
                         'validation_games', 'train_suite', 'test_suite']:
                if hasattr(args, attr):
                    setattr(self, attr, getattr(args, attr))
            
            self.hyperparams = vars(args).copy()
        else:
            # Legacy path for non-refactored scripts
            self.num_epochs = num_epochs
            self.hyperparams = hyperparams if hyperparams is not None else {}
            self.resume_from = resume_from
            self.use_best_checkpoint = use_best_checkpoint
            self.save_visualization_data = save_visualization_data
            
            # Bind extra kwargs for legacy classes
            for k, v in kwargs.items():
                setattr(self, k, v)
        
        # Initialize logger
        self.logger = TensorBoardLogger(
            training_type=training_type,
            pretrained_model_path=resume_from,
            hyperparams=hyperparams
        )
        
        # Training state
        self.start_epoch = 0
        self.best_metric = 0
        self.model = None
        self.optimizer = None
        self.writer = None
        self.log_dir = None
        
        # Visualization support
        self.save_visualization_data = save_visualization_data
        self.visualizer = None
        
    def setup(self):
        """
        Setup training: create model, optimizer, load checkpoint, setup TensorBoard.
        """
        # Print header
        self.logger.print_header()
        
        # Create model and optimizer
        self.model = self.create_model()
        self.optimizer = self.create_optimizer(self.model)
        
        # Load checkpoint if provided
        self.start_epoch, self.best_metric = self.logger.load_checkpoint(
            self.model, 
            self.optimizer,
            use_best_checkpoint=self.use_best_checkpoint
        )
        
        # Setup TensorBoard
        self.writer, self.log_dir, is_resuming = self.logger.setup_tensorboard()
        
        # Allow subclasses to do additional setup
        self.post_setup()
        
    def train(self):
        """
        Main training loop (template method).
        """
        # Compute baseline metrics if resuming
        if self.start_epoch > 0 or self.resume_from:
            self._compute_baseline_metrics()
        
        # Training loop
        end_epoch = self.start_epoch + self.num_epochs
        pbar = tqdm(
            range(self.start_epoch, end_epoch),
            desc="Training",
            unit="epoch",
            initial=self.start_epoch,
            total=end_epoch
        )
        
        for epoch in pbar:
            # Train one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Log all metrics
            all_metrics = {**train_metrics, **val_metrics}
            self.logger.log_scalars(all_metrics, epoch)
            
            # Update progress bar
            pbar.set_postfix(self.get_progress_bar_dict(train_metrics, val_metrics))
            
            # Periodic actions (e.g., render validation game)
            self.on_epoch_end(epoch, pbar)
            
            # Save checkpoint
            metric_value, metric_name = self.get_metric_for_checkpoint(val_metrics)
            is_best = metric_value > self.best_metric
            if is_best:
                self.best_metric = metric_value
            
            self.logger.save_checkpoint(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                metric_value=metric_value,
                metric_name=metric_name,
                is_best=is_best,
                additional_data=self.get_additional_checkpoint_data()
            )
        
        # Training complete
        self._finish_training()
    
    def _compute_baseline_metrics(self):
        """Compute and log baseline metrics when resuming from checkpoint."""
        print("=" * 60)
        print("Computing epoch 0 baseline metrics...")
        print("=" * 60)
        
        baseline_metrics = self.validate(epoch=0)
        self.logger.log_scalars(baseline_metrics, 0)
        
        # Print baseline
        for key, value in baseline_metrics.items():
            print(f"{key}: {value}")
        print("=" * 60 + "\n")
    
    def _finish_training(self):
        """Finish training: close logger, print summary."""
        self.logger.close()
        
        # Get final summary from subclass
        summary = self.get_final_summary()
        summary['Best Metric'] = f"{self.best_metric:.3f}"
        
        self.logger.print_completion_summary(summary)
    
    # ========== Abstract Methods (must be implemented by subclasses) ==========
    
    @abstractmethod
    def create_model(self) -> nn.Module:
        """
        Create and return the model.
        
        Returns:
            PyTorch model
        """
        pass
    
    @abstractmethod
    def create_optimizer(self, model: nn.Module) -> Optional[Optimizer]:
        """
        Create and return the optimizer.
        
        Args:
            model: The model to optimize
            
        Returns:
            PyTorch optimizer (or None for methods like Q-learning that don't use optimizers)
        """
        pass
    
    @abstractmethod
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics (e.g., {'Loss/train': 0.5, 'Performance/score': 100})
        """
        pass
    
    @abstractmethod
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Run validation.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics (e.g., {'Score/validation': 150, 'Score/win_rate': 0.8})
        """
        pass
    
    @abstractmethod
    def get_metric_for_checkpoint(self, val_metrics: Dict[str, float]) -> Tuple[float, str]:
        """
        Extract the metric to use for determining best checkpoint.
        
        Args:
            val_metrics: Dictionary of validation metrics
            
        Returns:
            Tuple of (metric_value, metric_name) for checkpoint tracking
        """
        pass
    
    # ========== Optional Methods (can be overridden by subclasses) ==========
    
    def post_setup(self):
        """
        Called after model/optimizer creation and checkpoint loading.
        Override to add custom setup logic.
        """
        pass
    
    def get_additional_checkpoint_data(self) -> Dict[str, Any]:
        """
        Get additional data to save in checkpoint.
        Override to add custom checkpoint data.
        
        Returns:
            Dictionary of additional data
        """
        return {}
    
    def get_progress_bar_dict(self, train_metrics: Dict, val_metrics: Dict) -> Dict:
        """
        Get dictionary for progress bar display.
        Override to customize progress bar.
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
            
        Returns:
            Dictionary of key-value pairs to display
        """
        # Default: show first 3 metrics
        all_metrics = {**train_metrics, **val_metrics}
        items = list(all_metrics.items())[:3]
        return {k.split('/')[-1]: f"{v:.3f}" for k, v in items}
    
    def on_epoch_end(self, epoch: int, pbar: tqdm):
        """
        Called at the end of each epoch.
        Override to add custom logic (e.g., render validation game).
        
        Args:
            epoch: Current epoch number
            pbar: Progress bar (for tqdm.write)
        """
        pass
    
    def get_final_summary(self) -> Dict[str, str]:
        """
        Get final summary to print at end of training.
        Override to add custom summary information.
        
        Returns:
            Dictionary of summary information
        """
        return {}
    
    # ========== Visualization Hooks (optional to override) ==========
    
    def create_visualizer(self):
        """
        Create and return a visualizer instance (e.g., EpochVisualizer).
        Override this method to enable training visualization.
        
        Example:
            from reinforcement_learning.training_visualization.epoch_visualizer import EpochVisualizer
            vis_dir = os.path.join(os.path.dirname(__file__), 'visualization_data')
            os.makedirs(vis_dir, exist_ok=True)
            return EpochVisualizer(vis_dir, self.hyperparams)
        
        Returns:
            Visualizer instance or None if visualization not needed
        """
        return None
    
    def on_visualization_epoch_start(self, epoch: int, batch_size: int):
        """
        Called at the start of each training epoch if visualization is enabled.
        Override to initialize epoch-specific visualization data.
        
        Args:
            epoch: Current epoch number
            batch_size: Batch size (number of parallel environments or size of training batch)
        """
        if self.visualizer:
            self.visualizer.start_epoch(epoch, batch_size)
    
    def on_visualization_step(self, env_idx: int, step_data: Dict[str, Any]):
        """
        Called during training to record a single step if visualization is enabled.
        Override to customize what data is recorded.
        
        Args:
            env_idx: Index of the environment/batch element
            step_data: Dictionary containing step information (state, action, reward, etc.)
        """
        if self.visualizer:
            self.visualizer.record_step(env_idx, step_data)
    
    def on_visualization_advantages(self, env_idx: int, advantages: list):
        """
        Called after advantage calculation if visualization is enabled.
        Override to record advantage values for analysis.
        
        Args:
            env_idx: Index of the environment/batch element
            advantages: List of advantage values
        """
        if self.visualizer:
            self.visualizer.record_advantages(env_idx, advantages)

    def on_visualization_batch_losses(self, actor_loss: torch.Tensor, critic_loss: torch.Tensor, 
                                all_entropies: torch.Tensor, total_loss: torch.Tensor, 
                                batch_size: int, steps_per_epoch: int):
        """
        Called after loss calculation if visualization is enabled.
        Override to record loss values for analysis.
        
        Args:
            actor_loss: Actor loss values
            critic_loss: Critic loss values
            all_entropies: Entropy values
            total_loss: Total loss values
            batch_size: Batch size
            steps_per_epoch: Number of steps per epoch
        """
        if self.visualizer:
            self.visualizer.record_batch_losses(actor_loss, critic_loss, all_entropies, total_loss, batch_size, steps_per_epoch)
    
    
    def on_visualization_epoch_end(self):
        """
        Called at the end of each training epoch if visualization is enabled.
        Override to finalize and save epoch visualization data.
        """
        if self.visualizer:
            self.visualizer.end_epoch()

