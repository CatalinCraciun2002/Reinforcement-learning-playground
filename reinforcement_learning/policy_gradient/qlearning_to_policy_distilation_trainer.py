U"""
Knowledge Distillation Trainer: Q-Learning to Policy Gradient

Trains a Policy Gradient neural network (student) by distilling knowledge
from a trained Q-learning linear model (teacher).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import os
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from reinforcement_learning.base_trainer import BaseTrainer
from runs.logger import TensorBoardLogger
from models.policy_gradient_models.simple_residual_conv import ActorCriticNetwork
from agents.qlearning_agents.qlearning_agent import ApproximateQAgent
from core.environment import PacmanEnv
from agents.policy_gradient_agents.deepRlAgent import RLAgent
import core.layout as layout_module
from display import graphicsDisplay


class DistillationTrainer(BaseTrainer):
    """
    Knowledge Distillation Trainer: transfers knowledge from Q-learning to Policy Gradient.
    
    Teacher: Q-learning agent with linear feature approximation
    Student: ActorCriticNetwork (Policy Gradient)
    Loss: KL divergence + optional value matching
    """
    
    def __init__(
        self,
        teacher_checkpoint,
        num_epochs=100,
        batch_size=16,
        steps_per_epoch=20,
        layout_name='mediumClassic',
        temperature=2.0,
        beta=0.5,
        lr=1e-4,
        memory_context=5,
        show_epochs=50,
        resume_from=None,
        use_best_checkpoint=False
    ):
        """
        Initialize Distillation Trainer.
        
        Args:
            teacher_checkpoint: Path to Q-learning checkpoint
            num_epochs: Number of training epochs
            batch_size: Number of parallel environments
            steps_per_epoch: Steps per environment per epoch
            layout_name: Pacman layout to use
            temperature: Softmax temperature for teacher (higher = softer)
            beta: Weight for value loss (0 = only policy distillation)
            lr: Learning rate
            memory_context: Memory context for student model
            show_epochs: Render validation game every N epochs (0 to disable)
            resume_from: Path to resume student checkpoint
            use_best_checkpoint: Load best checkpoint instead of last
        """
        self.teacher_checkpoint = teacher_checkpoint
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.layout_name = layout_name
        self.temperature = temperature
        self.beta = beta
        self.lr = lr
        self.memory_context = memory_context
        self.show_epochs = show_epochs
        self.use_best_checkpoint = use_best_checkpoint
        
        # Teacher and student
        self.teacher = None
        self.student_agent = None
        self.envs = None
        
        # Metrics
        self.total_steps = 0
        
        # Initialize parent with corrected parameters
        hyperparams_dict = {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'steps_per_epoch': steps_per_epoch,
            'layout': layout_name,
            'temperature': temperature,
            'beta': beta,
            'learning_rate': lr,
            'memory_context': memory_context,
        }
        
        super().__init__(
            training_type='distillation',
            num_epochs=num_epochs,
            hyperparams=hyperparams_dict,
            resume_from=resume_from,
            use_best_checkpoint=use_best_checkpoint
        )
    
    def create_model(self):
        """Create student Policy Gradient network."""
        return ActorCriticNetwork(memory_context=self.memory_context)
    
    def create_optimizer(self, model):
        """Create Adam optimizer for student."""
        return optim.Adam(model.parameters(), lr=self.lr)
    
    def post_setup(self):
        """Load teacher and setup environments after model/optimizer creation."""
        # Load teacher Q-learning agent
        print(f"\nLoading teacher Q-learning model from: {self.teacher_checkpoint}")
        self.teacher = self._load_teacher()
        print(f"✓ Teacher loaded successfully")
        print(f"  Teacher weights: {len(self.teacher.weights)} features")
        
        # Create student agent
        self.student_agent = RLAgent(self.model, memory_context=self.memory_context)
        
        # Create environments with proper env_id tracking
        self.envs = [
            PacmanEnv(self.student_agent, self.layout_name, env_id=i) 
            for i in range(self.batch_size)
        ]
        print(f"✓ Created {self.batch_size} parallel environments")
    
    def _load_teacher(self):
        """Load Q-learning teacher from checkpoint."""
        if not os.path.exists(self.teacher_checkpoint):
            raise FileNotFoundError(f"Teacher checkpoint not found: {self.teacher_checkpoint}")
        
        # Determine checkpoint path
        if os.path.isdir(self.teacher_checkpoint):
            checkpoint_path = os.path.join(self.teacher_checkpoint, 'model_last.pth')
        else:
            checkpoint_path = self.teacher_checkpoint
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Create teacher agent
        teacher = ApproximateQAgent()
        
        # Load weights
        if 'qlearning_weights' in checkpoint:
            teacher.weights = checkpoint['qlearning_weights']
        elif 'model_state_dict' in checkpoint:
            # Fallback: try to extract from model state dict (shouldn't happen)
            raise ValueError("Checkpoint doesn't contain Q-learning weights")
        else:
            raise ValueError("Unknown checkpoint format")
        
        # Set to evaluation mode (no exploration)
        teacher.epsilon = 0.0
        
        return teacher
    
    def _get_teacher_probs(self, state):
        """
        Get teacher's action probabilities using softmax over Q-values.
        
        Args:
            state: Game state
            
        Returns:
            action_probs: Tensor of shape (5,) with probabilities for each action
            max_q_value: Maximum Q-value (for value distillation)
        """
        legal_actions = state.getLegalPacmanActions()
        
        # Get Q-values for all actions
        from core.game import Directions
        all_actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        action_to_idx = {action: idx for idx, action in enumerate(all_actions)}
        
        q_values = []
        for action in all_actions:
            if action in legal_actions:
                q_val = self.teacher.getQValue(state, action)
            else:
                q_val = float('-inf')  # Invalid actions get -inf
            q_values.append(q_val)
        
        q_values = torch.tensor(q_values, dtype=torch.float32)
        max_q_value = q_values[q_values != float('-inf')].max().item() if len(q_values[q_values != float('-inf')]) > 0 else 0.0
        
        # Apply temperature and softmax
        q_values_temp = q_values / self.temperature
        
        # Mask illegal actions before softmax
        mask = torch.tensor([action in legal_actions for action in all_actions], dtype=torch.bool)
        q_values_temp[~mask] = float('-inf')
        
        action_probs = F.softmax(q_values_temp, dim=0)
        
        return action_probs, max_q_value
    
    def train_epoch(self, epoch):
        """
        Train for one epoch by collecting data and distilling knowledge.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            dict: Training metrics
        """
        self.model.train()
        
        # Metrics
        epoch_kl_loss = 0.0
        epoch_value_loss = 0.0
        epoch_total_loss = 0.0
        num_steps = 0
        
        # Reset environments and initialize agents
        states = []
        for env_idx, env in enumerate(self.envs):
            state = env.reset()
            states.append(state)
        
        # Collect data and train
        for step in range(self.steps_per_epoch):
            # Get teacher and student predictions for all environments
            teacher_probs_batch = []
            teacher_max_q_batch = []
            student_states_batch = []
            
            for env_idx, (env, state) in enumerate(zip(self.envs, states)):
                # Get current state tensor
                state_tensor = self.student_agent.state_to_tensor(state, env_id=env_idx)
                student_states_batch.append(state_tensor)
                
                # Get teacher probabilities
                teacher_probs, max_q = self._get_teacher_probs(state)
                teacher_probs_batch.append(teacher_probs)
                teacher_max_q_batch.append(max_q)
            
            # Stack batch - each state_tensor already has batch dim
            student_states_batch = torch.cat(student_states_batch, dim=0)
            teacher_probs_batch = torch.stack(teacher_probs_batch)
            teacher_max_q_batch = torch.tensor(teacher_max_q_batch, dtype=torch.float32).unsqueeze(1)
            
            # Get student predictions
            student_probs, student_values = self.model(student_states_batch, return_both=True)
            
            # Compute KL divergence loss (main distillation loss)
            # KL(teacher || student) = sum(teacher * log(teacher / student))
            kl_loss = F.kl_div(
                student_probs.log(),
                teacher_probs_batch,
                reduction='batchmean',
                log_target=False
            )
            
            # Compute value loss (optional)
            value_loss = F.mse_loss(student_values, teacher_max_q_batch)
            
            # Total loss
            total_loss = kl_loss + self.beta * value_loss
            
            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update metrics
            epoch_kl_loss += kl_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_total_loss += total_loss.item()
            num_steps += 1
            self.total_steps += self.batch_size
            
            # Take actions in environments (use student for exploration)
            new_states = []
            for env_idx, (env, state) in enumerate(zip(self.envs, states)):
                # Get action probabilities from student
                probs, _ = self.student_agent.forward(state, env_id=env_idx)
                legal_actions = state.getLegalPacmanActions()
                action, _ = self.student_agent.getAction(legal_actions, probs)
                
                next_state, reward, done = env.step(action)
                
                if done:
                    next_state = env.reset()
                
                new_states.append(next_state)
            
            states = new_states
        
        # Average metrics
        metrics = {
            'kl_loss': epoch_kl_loss / num_steps,
            'value_loss': epoch_value_loss / num_steps,
            'total_loss': epoch_total_loss / num_steps,
        }
        
        return metrics
    
    def validate(self, epoch):
        """
        Run validation games.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        
        validation_games = 8
        total_score = 0
        wins = 0
        
        with torch.no_grad():
            for _ in range(validation_games):
                # Create validation environment with unique env_id
                val_env_id = self.batch_size + _  # Use IDs beyond training envs
                env = PacmanEnv(self.student_agent, self.layout_name, env_id=val_env_id)
                state = env.reset()
                done = False
                
                while not done:
                    probs, _ = self.student_agent.forward(state, env_id=val_env_id)
                    legal_actions = state.getLegalPacmanActions()
                    action, _ = self.student_agent.getAction(legal_actions, probs)
                    state, reward, done = env.step(action)
                
                # Get final score and outcome from the environment
                total_score += env.game.state.getScore()
                if env.game.state.isWin():
                    wins += 1
        
        avg_score = total_score / validation_games
        win_rate = wins / validation_games
        
        return {
            'val_score': avg_score,
            'val_win_rate': win_rate,
        }
    
    def get_metric_for_checkpoint(self, val_metrics):
        """Use validation score for checkpointing."""
        return val_metrics['val_score'], 'val_score'
    
    def get_progress_bar_dict(self, train_metrics, val_metrics):
        """Customize progress bar display."""
        return {
            'KL': f"{train_metrics['kl_loss']:.4f}",
            'Value': f"{train_metrics['value_loss']:.4f}",
            'ValScore': f"{val_metrics['val_score']:.1f}",
        }
    
    def on_epoch_end(self, epoch, pbar):
        """Display student game with graphics at specified intervals."""
        if self.show_epochs > 0 and (epoch + 1) % self.show_epochs == 0:
            pbar.write(f"\n{'='*60}")
            pbar.write(f"Running student game with graphics at epoch {epoch+1}...")
            pbar.write('='*60)
            
            self.model.eval()
            
            # Create environment with graphics
            display = graphicsDisplay.PacmanGraphics(1.0, frameTime=0.05)
            val_env = PacmanEnv(self.student_agent, self.layout_name, display=display, env_id=self.batch_size + 100)
            state = val_env.reset()
            done = False
            steps = 0
            max_steps = 1000
            
            with torch.no_grad():
                while not done and steps < max_steps:
                    probs, _ = self.student_agent.forward(state, env_id=self.batch_size + 100)
                    legal_actions = state.getLegalPacmanActions()
                    action, _ = self.student_agent.getAction(legal_actions, probs)
                    state, reward, done = val_env.step(action)
                    steps += 1
            
            score = val_env.game.state.getScore()
            won = val_env.game.state.isWin()
            
            pbar.write(f"\nStudent Game Result - Score: {score}, {'WON!' if won else 'Lost'}, Steps: {steps}")
            pbar.write('='*60 + '\n')
            
            self.model.train()
    
    def get_final_summary(self):
        """Get final training summary."""
        return {
            'Total Steps': self.total_steps,
            'Best Val Score': f"{self.best_metric:.1f}",
        }


def main():
    parser = argparse.ArgumentParser(description='Q-Learning to Policy Gradient Knowledge Distillation')
    parser.add_argument('--teacher-checkpoint', type=str, default='runs\qlearning\\20260211_125233',
                       help='Path to Q-learning teacher checkpoint')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Number of parallel environments')
    parser.add_argument('--steps-per-epoch', type=int, default=20,
                       help='Steps per environment per epoch')
    parser.add_argument('--layout', type=str, default='mediumClassic',
                       help='Layout name')
    parser.add_argument('--temperature', type=float, default=2.0,
                       help='Softmax temperature for teacher (higher = softer)')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='Weight for value loss (0 = only policy distillation)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--memory-context', type=int, default=5,
                       help='Memory context size')
    parser.add_argument('--show-epochs', type=int, default=10,
                       help='Render student game every N epochs (0 to disable)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to student checkpoint to resume from')
    parser.add_argument('--use-best', action='store_true', default=True,
                       help='Load best checkpoint instead of last')
    
    args = parser.parse_args()
    
    # Create and run trainer
    trainer = DistillationTrainer(
        teacher_checkpoint=args.teacher_checkpoint,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        layout_name=args.layout,
        temperature=args.temperature,
        beta=args.beta,
        lr=args.lr,
        memory_context=args.memory_context,
        show_epochs=args.show_epochs,
        resume_from=args.resume,
        use_best_checkpoint=args.use_best
    )
    
    trainer.setup()
    trainer.train()


if __name__ == '__main__':
    main()
