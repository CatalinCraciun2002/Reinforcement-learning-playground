"""
Train ActorCriticNetwork using Human Gameplay Recordings - Refactored with BaseTrainer

This script implements supervised learning (behavioral cloning) to train
the ActorCriticNetwork model to imitate human gameplay.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

from reinforcement_learning.base_trainer import BaseTrainer
from models.policy_gradient_models.simple_residual_conv import ActorCriticNetwork
from human_feedback.data_loader import GameplayDataset
from agents.policy_gradient_agents.deepRlAgent import RLAgent
from core.game_orchestrator import GameOrchestrator


# Action mapping: string to index
ACTION_TO_IDX = {
    'North': 0,
    'South': 1,
    'East': 2,
    'West': 3,
    'Stop': 4  # Stop is a separate action
}


def compute_discounted_returns(episode_transitions, gamma=0.99):
    """
    Compute discounted returns for an episode.
    
    Args:
        episode_transitions: List of transitions from a single episode
        gamma: Discount factor
    
    Returns:
        List of discounted returns for each transition
    """
    returns = []
    G = 0
    
    for transition in reversed(episode_transitions):
        G = transition['reward'] + gamma * G
        returns.insert(0, G)
    
    return returns


def prepare_training_data(dataset, memory_length=5, gamma=0.99):
    """
    Prepare training data from all episodes with returns computed.
    
    Args:
        dataset: GameplayDataset instance
        memory_length: Number of past positions to include
        gamma: Discount factor for computing returns
    
    Returns:
        (train_data, episode_scores) where:
        - train_data: List of dicts with state_tensor, action_idx, return_value
        - episode_scores: List of final scores from each episode
    """
    all_data = []
    episode_scores = []
    
    print("Preparing training data from all games...")
    for episode in dataset.episodes:
        walls = episode['walls']
        transitions = episode['transitions']
        
        # Get final score from last transition
        if transitions:
            final_score = transitions[-1]['state'].get('score', 0)
            episode_scores.append(final_score)
        
        # Compute returns for this episode
        returns = compute_discounted_returns(transitions, gamma)
        
        # Process each transition
        for i, (transition, G) in enumerate(zip(transitions, returns)):
            # Gather past positions
            past_positions = []
            for j in range(max(0, i - memory_length), i):
                past_positions.append(transitions[j]['state']['pacman_pos'])
            
            # Convert state to tensor
            state_tensor = dataset.state_to_tensor(
                transition['state'],
                walls,
                past_positions=past_positions,
                memory_length=memory_length
            )
            
            # Convert action to index
            action_str = transition['action']
            action_idx = ACTION_TO_IDX.get(action_str, 0)
            
            all_data.append({
                'state': state_tensor,
                'action': action_idx,
                'return': G
            })
    
    print(f"Training samples: {len(all_data)} from {len(episode_scores)} episodes")
    print(f"Average human score: {np.mean(episode_scores):.1f}")
    
    return all_data, episode_scores


class HumanFeedbackTrainer(BaseTrainer):
    """Human Feedback Trainer using supervised learning and BaseTrainer framework."""
    
    def __init__(
        self,
        data_dir='game_runs_data',
        num_epochs=50,
        batch_size=32,
        lr=1e-4,
        memory_length=5,
        train_critic=True,
        gamma=0.99,
        train_suite='standard_only',
        test_suite='standard_only',
        validation_games=8,
        resume_from=None
    ):
        """
        Initialize Human Feedback Trainer.
        
        Args:
            data_dir: Directory containing gameplay recordings
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            memory_length: Number of past positions to include
            train_critic: Whether to train the critic head
            gamma: Discount factor for computing returns
            layout_name: Layout to use for validation games
            validation_games: Number of validation games to play each epoch
            resume_from: Path to checkpoint to resume from
        """
        self.data_dir = data_dir
        self.batch_size_train = batch_size
        self.memory_length = memory_length
        self.train_critic = train_critic
        self.gamma = gamma
        self.train_suite = train_suite
        self.test_suite = test_suite
        self.validation_games = validation_games
        self.lr = lr

        self.train_data = None
        self.episode_scores = None
        self.agent = None
        self.orchestrator = None
        
        hyperparams = {
            'batch_size': batch_size,
            'learning_rate': lr,
            'memory_length': memory_length,
            'gamma': gamma,
            'train_critic': train_critic,
            'num_epochs': num_epochs,
            'train_suite': train_suite,
            'test_suite': test_suite,
        }
        
        # Initialize base class
        super().__init__(
            training_type='human_feedback',
            num_epochs=num_epochs,
            hyperparams=hyperparams,
            resume_from=resume_from,
            use_best_checkpoint=False
        )
    
    def create_model(self):
        """Create Actor-Critic network."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}\n")
        model = ActorCriticNetwork(memory_context=self.memory_length)
        return model.to(device)
    
    def create_optimizer(self, model):
        """Create Adam optimizer."""
        return optim.Adam(model.parameters(), lr=self.lr)
    
    def post_setup(self):
        """Load and prepare training data, create agent and orchestrator."""
        dataset = GameplayDataset(self.data_dir)

        if len(dataset.episodes) == 0:
            raise ValueError("Error: No gameplay recordings found!")

        dataset.print_statistics()

        self.train_data, self.episode_scores = prepare_training_data(
            dataset, self.memory_length, gamma=self.gamma
        )

        self.agent = RLAgent(self.model, memory_context=self.memory_length)
        self.orchestrator = GameOrchestrator(
            agent=self.agent,
            batch_size=1,
            train_suite_name=self.train_suite,
            test_suite_name=self.test_suite,
        )
    
    def train_epoch(self, epoch):
        """Train for one epoch using supervised learning."""
        self.model.train()
        
        # Shuffle data
        np.random.shuffle(self.train_data)
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_correct = 0
        num_batches = 0
        
        device = next(self.model.parameters()).device
        
        for i in range(0, len(self.train_data), self.batch_size_train):
            batch = self.train_data[i:i + self.batch_size_train]
            
            # Prepare batch tensors
            states = torch.stack([item['state'] for item in batch]).to(device)
            actions = torch.tensor([item['action'] for item in batch], dtype=torch.long).to(device)
            returns = torch.tensor([item['return'] for item in batch], dtype=torch.float32).to(device)
            
            # Forward pass
            action_probs, values = self.model(states, return_both=True)
            
            # Actor loss (cross-entropy)
            actor_loss = nn.CrossEntropyLoss()(action_probs, actions)
            
            # Critic loss (MSE with normalized returns)
            if self.train_critic:
                return_mean = returns.mean().detach()
                return_std = returns.std().detach() + 1e-8
                values_norm = (values.squeeze() - return_mean) / return_std
                returns_norm = (returns - return_mean) / return_std
                critic_loss = nn.MSELoss()(values_norm, returns_norm)
            else:
                critic_loss = torch.tensor(0.0)
            
            # Total loss with coefficients
            total_loss = 1.0 * actor_loss + 0.5 * critic_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item() if self.train_critic else 0
            
            predicted_actions = torch.argmax(action_probs, dim=1)
            total_correct += (predicted_actions == actions).sum().item()
            
            num_batches += 1
        
        avg_actor_loss = total_actor_loss / num_batches
        avg_critic_loss = total_critic_loss / num_batches if self.train_critic else 0
        accuracy = total_correct / len(self.train_data)
        
        return {
            'Loss/train_actor': avg_actor_loss,
            'Loss/train_critic': avg_critic_loss,
            'Accuracy/train': accuracy
        }
    
    def validate(self, epoch):
        """Validate by playing games via the orchestrator."""
        self.model.eval()

        results = self.orchestrator.run_validation(
            n_games=self.validation_games, with_graphics=False
        )
        scores = [r[0] for r in results]
        avg_score = np.mean(scores)

        return {
            'Score/validation': avg_score
        }
    
    def get_metric_for_checkpoint(self, val_metrics):
        """Use validation score as the metric for checkpointing."""
        return val_metrics['Score/validation'], 'val_score'
    
    def get_progress_bar_dict(self, train_metrics, val_metrics):
        """Customize progress bar display."""
        return {
            'TrainLoss': f"{train_metrics.get('Loss/train_actor', 0):.3f}",
            'TrainAcc': f"{train_metrics.get('Accuracy/train', 0):.3f}",
            'ValScore': f"{val_metrics.get('Score/validation', 0):.0f}"
        }
    
    def get_final_summary(self):
        """Get final training summary."""
        return {
            'Best Validation Score': f"{self.best_metric:.0f}"
        }


def main():
    parser = argparse.ArgumentParser(description='Train ActorCriticNetwork from human gameplay')
    parser.add_argument('--data-dir', type=str, default='game_runs_data',
                       help='Directory containing gameplay recordings')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--memory-length', type=int, default=5,
                       help='Number of past positions to include')
    parser.add_argument('--no-critic', action='store_true',
                       help='Do not train the critic head')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor for computing returns')
    parser.add_argument('--train-suite', type=str, default='standard_only',
                       help='Scenario suite name for training')
    parser.add_argument('--test-suite', type=str, default='standard_only',
                       help='Scenario suite name for validation')
    parser.add_argument('--validation-games', type=int, default=8,
                       help='Number of validation games to play each epoch')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint or run directory to resume from')
    
    args = parser.parse_args()
    
    trainer = HumanFeedbackTrainer(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        memory_length=args.memory_length,
        train_critic=not args.no_critic,
        gamma=args.gamma,
        train_suite=args.train_suite,
        test_suite=args.test_suite,
        validation_games=args.validation_games,
        resume_from=args.resume
    )
    
    trainer.setup()
    trainer.train()


if __name__ == '__main__':
    main()
