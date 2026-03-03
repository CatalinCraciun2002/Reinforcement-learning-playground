"""
DQN Training Script for Pacman - Refactored with BaseTrainer
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import os
import argparse

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from reinforcement_learning.base_trainer import BaseTrainer
from agents.deep_qlearning_agents.dqn_agent import DQNAgent, Directions
from models.deep_qlearning_models.dqn_model import DQN
import core.layout as layout_module
from core.pacman import ClassicGameRules
import display.textDisplay as textDisplay
import display.graphicsDisplay as graphicsDisplay
import agents.base_agents.ghostAgents as ghostAgents


class DQNTrainer(BaseTrainer):
    """DQN Trainer using BaseTrainer framework."""
    
    def __init__(
        self,
        num_epochs=1000,
        layout_name='mediumClassic',
        max_steps=500,
        update_target_every=10,
        render_every=0,
        view_speed=0.05,
        lr=1e-4,
        resume_from=None
    ):
        """
        Initialize DQN Trainer.
        
        Args:
            num_epochs: Number of training episodes
            layout_name: Layout to train on
            max_steps: Max steps per episode
            update_target_every: Update target network every N episodes
            render_every: Render validation game every N episodes (0 to disable)
            view_speed: Speed of rendered game
            lr: Learning rate
            resume_from: Path to checkpoint to resume from
        """
        # Store DQN-specific hyperparameters
        self.layout_name = layout_name
        self.max_steps = max_steps
        self.update_target_every = update_target_every
        self.render_every = render_every
        self.view_speed = view_speed
        self.lr = lr
        
        # Hyperparameters for logging
        hyperparams = {
            'num_epochs': num_epochs,
            'layout': layout_name,
            'max_steps': max_steps,
            'learning_rate': lr,
            'update_target_every': update_target_every
        }
        
        # Initialize base class
        super().__init__(
            training_type='dqn',
            num_epochs=num_epochs,
            hyperparams=hyperparams,
            resume_from=resume_from,
            use_best_checkpoint=False
        )
        
        # DQN-specific tracking
        self.agent = None
        self.layout = None
        self.ghosts = None
        self.rules = None
        self.loss_fn = None
        
        self.total_wins = 0
        self.all_scores = []
        self.all_losses = []
    
    def create_model(self):
        """Create DQN model."""
        # Note: We return a placeholder here because DQN is inside DQNAgent
        # The actual model is created in post_setup
        return DQN(num_frames=50)
    
    def create_optimizer(self, model):
        """Create optimizer for DQN."""
        return optim.Adam(model.parameters(), lr=self.lr)
    
    def post_setup(self):
        """Setup DQN agent and environment after model/optimizer creation."""
        # Create agent (which contains model and target_model)
        self.agent = DQNAgent(epsilon=1.0)
        
        # Replace agent's model with our loaded model
        self.agent.model = self.model
        
        # Update target model to match
        self.agent.target_model.load_state_dict(self.model.state_dict())
        
        # Setup environment
        self.layout = layout_module.getLayout(self.layout_name)
        self.ghosts = [ghostAgents.RandomGhost(i+1) for i in range(4)]
        self.rules = ClassicGameRules()
        
        # Loss function
        self.loss_fn = torch.nn.SmoothL1Loss()
    
    def train_epoch(self, epoch):
        """Train for one episode (DQN trains per-episode)."""
        # Create game
        game = self.rules.newGame(
            self.layout,
            self.agent,
            self.ghosts,
            textDisplay.NullGraphics(),
            quiet=True,
            catchExceptions=False
        )
        self.agent.registerInitialState(game.state)
        
        state_tensor = self.agent.get_temporal_input(game.state)
        
        episode_losses = []
        episode_q_values = []
        steps = 0
        
        while not game.gameOver and steps < self.max_steps:
            steps += 1
            
            # Get action
            action = self.agent.getAction(game.state)
            action_idx = self.agent.action_to_idx.get(action, 0)
            
            prev_score = game.state.getScore()
            prev_state_tensor = state_tensor.clone()
            
            # Step environment
            game.state = game.state.generateSuccessor(0, action)
            game.display.update(game.state.data)
            game.rules.process(game.state, game)
            
            # Compute reward
            new_score = game.state.getScore()
            reward = new_score - prev_score
            
            done = game.gameOver
            if done and game.state.isWin():
                reward += 100
                self.total_wins += 1
            
            if action != Directions.STOP:
                reward += 0.1
            
            # Get next state
            if done:
                next_state_tensor = None
            else:
                next_state_tensor = self.agent.get_temporal_input(game.state)
                state_tensor = next_state_tensor
            
            # Store in replay buffer
            self.agent.memory.push(
                prev_state_tensor.cpu(),
                action_idx,
                reward,
                next_state_tensor.cpu() if next_state_tensor is not None else None,
                done
            )
            
            # Train from replay buffer
            if len(self.agent.memory) > self.agent.batch_size:
                loss, avg_q = self._update_model()
                episode_losses.append(loss)
                episode_q_values.append(avg_q)
        
        # Decay epsilon
        self.agent.decay_epsilon()
        
        # Update target network
        if epoch % self.update_target_every == 0:
            self.agent.target_model.load_state_dict(self.agent.model.state_dict())
        
        # Track metrics
        final_score = game.state.getScore()
        self.all_scores.append(final_score)
        if episode_losses:
            self.all_losses.extend(episode_losses)
        
        # Return training metrics
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        avg_q = np.mean(episode_q_values) if episode_q_values else 0
        
        return {
            'Loss/td_loss': avg_loss,
            'Performance/epsilon': self.agent.epsilon,
            'Performance/avg_q': avg_q,
            'Performance/steps': steps
        }
    
    def _update_model(self):
        """Update model from replay buffer."""
        transitions = self.agent.memory.sample(self.agent.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
        
        # Prepare batches
        batch_state = torch.cat(batch_state).to(self.model.conv1.weight.device)
        batch_action = torch.tensor(batch_action).unsqueeze(1).to(self.model.conv1.weight.device)
        batch_reward = torch.tensor(batch_reward).float().unsqueeze(1).to(self.model.conv1.weight.device)
        batch_done = torch.tensor(batch_done).float().unsqueeze(1).to(self.model.conv1.weight.device)
        
        # Compute Q(s, a)
        q_values = self.model(batch_state)
        current_q = q_values.gather(1, batch_action)
        avg_q = current_q.mean().item()
        
        # Compute target Q
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch_next_state)),
            dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch_next_state if s is not None])
        
        next_q_values = torch.zeros(self.agent.batch_size).to(self.model.conv1.weight.device)
        with torch.no_grad():
            if len(non_final_next_states) > 0:
                next_q_values[non_final_mask] = self.agent.target_model(non_final_next_states).max(1)[0]
        
        target_q = batch_reward + (0.99 * next_q_values.unsqueeze(1)) * (1 - batch_done)
        
        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item(), avg_q
    
    def validate(self, epoch):
        """Run validation (compute average score over recent episodes)."""
        # Use last 100 scores for validation metric
        recent_scores = self.all_scores[-100:] if self.all_scores else [0]
        avg_score = np.mean(recent_scores)
        
        return {
            'Score/avg_score': avg_score,
            'Score/wins': self.total_wins
        }
    
    def get_metric_for_checkpoint(self, val_metrics):
        """Use average score as the metric for checkpointing."""
        return val_metrics['Score/avg_score'], 'avg_score'
    
    def get_progress_bar_dict(self, train_metrics, val_metrics):
        """Customize progress bar display."""
        return {
            'Score': f"{val_metrics['Score/avg_score']:.1f}",
            'Wins': self.total_wins,
            'Q': f"{train_metrics.get('Performance/avg_q', 0):.1f}",
            'Loss': f"{train_metrics.get('Loss/td_loss', 0):.3f}"
        }
    
    def on_epoch_end(self, epoch, pbar):
        """Render validation game at specified intervals."""
        if self.render_every > 0 and (epoch + 1) % self.render_every == 0:
            pbar.write(f"\n{'='*60}")
            pbar.write(f"Running validation game with graphics at epoch {epoch+1}...")
            pbar.write('='*60)
            
            score, won = self._run_validation_game()
            
            pbar.write(f"Validation Result: Score={score}, {'WON!' if won else 'Lost'}")
            pbar.write('='*60 + '\n')
    
    def _run_validation_game(self):
        """Run a single validation game with graphics."""
        old_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.05  # Low epsilon for validation
        
        display = graphicsDisplay.PacmanGraphics(1.0, frameTime=self.view_speed)
        
        game = self.rules.newGame(
            self.layout,
            self.agent,
            self.ghosts,
            display,
            quiet=False,
            catchExceptions=False
        )
        
        game.run()
        
        self.agent.epsilon = old_epsilon
        
        return game.state.getScore(), game.state.isWin()
    
    def get_final_summary(self):
        """Get final training summary."""
        return {
            'Total Wins': f"{self.total_wins}",
            'Avg Score (last 100)': f"{np.mean(self.all_scores[-100:]):.1f}" if self.all_scores else "0"
        }
    
    def get_additional_checkpoint_data(self):
        """Save additional DQN-specific data."""
        return {
            'total_wins': self.total_wins,
            'epsilon': self.agent.epsilon,
            'replay_buffer_size': len(self.agent.memory)
        }


def main():
    parser = argparse.ArgumentParser(description='DQN Pacman Training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--layout', type=str, default='mediumClassic', help='Layout name')
    parser.add_argument('--max_steps', type=int, default=500, help='Max steps per episode')
    parser.add_argument('--update_target_every', type=int, default=10,
                       help='Update target network every N episodes')
    parser.add_argument('--render_every', type=int, default=50,
                       help='Render validation game every N episodes (0 to disable)')
    parser.add_argument('--view_speed', type=float, default=0.05, help='Speed of rendered game')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    trainer = DQNTrainer(
        num_epochs=args.epochs,
        layout_name=args.layout,
        max_steps=args.max_steps,
        update_target_every=args.update_target_every,
        render_every=args.render_every,
        view_speed=args.view_speed,
        lr=args.lr,
        resume_from=args.resume
    )
    
    trainer.setup()
    trainer.train()


if __name__ == '__main__':
    main()
