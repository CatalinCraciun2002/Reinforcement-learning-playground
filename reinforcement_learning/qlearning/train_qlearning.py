"""
Approximate Q-Learning Training Script - Refactored with BaseTrainer
"""
import sys
import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn

# Add project root to path (go up 3 levels: qlearning -> reinforcement_learning -> PacMan)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from reinforcement_learning.base_trainer import BaseTrainer
from agents.qlearning_agents.qlearning_agent import ApproximateQAgent
import core.layout as layout_module
from core.pacman import ClassicGameRules
import display.textDisplay as textDisplay
import display.graphicsDisplay as graphicsDisplay
import agents.base_agents.ghostAgents as ghostAgents
from core.game import Directions


class QLearningTrainer(BaseTrainer):
    """Q-Learning Trainer using BaseTrainer framework."""
    
    def __init__(
        self,
        num_episodes=100,
        layout_name='mediumClassic',
        alpha=0.2,
        gamma=0.8,
        epsilon=0.05,
        render_every=0,
        view_speed=0.05,
        validation_games=10,
        resume_from=None
    ):
        """
        Initialize Q-Learning Trainer.
        
        Args:
            num_episodes: Number of training episodes
            layout_name: Layout to train on
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            render_every: Render validation game every N episodes (0 to disable)
            view_speed: Speed of rendered game
            validation_games: Number of games to play for validation
            resume_from: Path to checkpoint to resume from
        """
        # Store Q-learning specific hyperparameters
        self.layout_name = layout_name
        self.alpha = alpha
        self.gamma = gamma
        self.initial_epsilon = epsilon
        self.render_every = render_every
        self.view_speed = view_speed
        self.validation_games = validation_games
        
        # Hyperparameters for logging
        hyperparams = {
            'num_episodes': num_episodes,
            'layout': layout_name,
            'alpha': alpha,
            'gamma': gamma,
            'epsilon': epsilon,
            'validation_games': validation_games
        }
        
        # Initialize base class
        super().__init__(
            training_type='qlearning',
            num_epochs=num_episodes,  # For Q-learning, epochs = episodes
            hyperparams=hyperparams,
            resume_from=resume_from,
            use_best_checkpoint=False
        )
        
        # Q-learning specific tracking
        self.agent = None
        self.layout = None
        self.ghosts = None
        self.rules = None
        
        self.total_wins = 0
        self.all_scores = []
    
    def create_model(self):
        """
        Create Q-learning agent.
        Note: Q-learning doesn't use a PyTorch model, so we return a dummy nn.Module.
        """
        # Q-learning uses feature weights, not a neural network
        # We return a dummy module to satisfy the base class interface
        class DummyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))
        
        return DummyModule()
    
    def create_optimizer(self, model):
        """Q-learning doesn't use an optimizer."""
        return None
    
    def post_setup(self):
        """Setup Q-learning agent and environment after base setup."""
        # Create agent
        self.agent = ApproximateQAgent(
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=self.initial_epsilon,
            numTraining=self.num_epochs
        )
        
        # Load weights from checkpoint if resuming
        if self.logger.checkpoint_data and 'qlearning_weights' in self.logger.checkpoint_data:
            self.agent.weights = self.logger.checkpoint_data['qlearning_weights']
            print(f"  ✓ Q-learning weights loaded from checkpoint")
        
        # Setup environment
        self.layout = layout_module.getLayout(self.layout_name)
        self.ghosts = [ghostAgents.DirectionalGhost(i+1) for i in range(4)]
        self.rules = ClassicGameRules()
    
    def train_epoch(self, epoch):
        """Train for one episode."""
        # Create game
        game = self.rules.newGame(
            self.layout,
            self.agent,
            self.ghosts,
            textDisplay.NullGraphics(),
            quiet=True,
            catchExceptions=False
        )
        game.display.initialize(game.state.data)
        
        # Run episode step-by-step to allow Q-learning updates
        while not game.gameOver:
            for i, agent_obj in enumerate(game.agents):
                if game.gameOver:
                    break
                
                # Get action
                state = game.state
                action = agent_obj.getAction(state)
                
                # Execute action
                game.state = game.state.generateSuccessor(i, action)
                game.display.update(game.state.data)
                game.rules.process(game.state, game)
                
                # Update Q-learning agent (only for Pacman, i=0)
                if i == 0:
                    next_state = game.state
                    reward = next_state.getScore() - state.getScore()
                    
                    # Win/lose bonus
                    if game.gameOver:
                        if game.state.isWin():
                            reward += 500
                            self.total_wins += 1
                        elif game.state.isLose():
                            reward -= 500
                    
                    # Update agent
                    if action != Directions.STOP:
                        reward += 0
                    self.agent.update(state, action, next_state, reward)
        
        # Track score
        final_score = game.state.getScore()
        self.all_scores.append(final_score)
        
        # Decay epsilon
        if self.agent.epsilon > 0.05:
            self.agent.epsilon *= 0.99
        
        # Return training metrics
        return {
            'Performance/epsilon': self.agent.epsilon,
            'Performance/score': final_score
        }
    
    def validate(self, epoch):
        """Run validation games."""
        val_scores = []
        val_wins = 0
        
        # Save training epsilon
        train_epsilon = self.agent.epsilon
        
        # Use low epsilon for validation (mostly exploit)
        self.agent.epsilon = 0.05
        
        for _ in range(self.validation_games):
            game = self.rules.newGame(
                self.layout,
                self.agent,
                self.ghosts,
                textDisplay.NullGraphics(),
                quiet=True,
                catchExceptions=False
            )
            game.display.initialize(game.state.data)
            
            # Run game
            while not game.gameOver:
                for i, agent_obj in enumerate(game.agents):
                    if game.gameOver:
                        break
                    action = agent_obj.getAction(game.state)
                    game.state = game.state.generateSuccessor(i, action)
                    game.display.update(game.state.data)
                    game.rules.process(game.state, game)
            
            val_scores.append(game.state.getScore())
            if game.state.isWin():
                val_wins += 1
        
        # Restore training epsilon
        self.agent.epsilon = train_epsilon
        
        avg_score = np.mean(val_scores)
        win_rate = val_wins / self.validation_games
        
        return {
            'Score/validation': avg_score,
            'Score/win_rate': win_rate
        }
    
    def get_metric_for_checkpoint(self, val_metrics):
        """Use validation score as the metric for checkpointing."""
        return val_metrics['Score/validation'], 'val_score'
    
    def get_progress_bar_dict(self, train_metrics, val_metrics):
        """Customize progress bar display."""
        recent_scores = self.all_scores[-100:] if len(self.all_scores) >= 100 else self.all_scores
        avg_score = np.mean(recent_scores) if recent_scores else 0
        
        return {
            'AvgScore': f"{avg_score:.1f}",
            'ValScore': f"{val_metrics.get('Score/validation', 0):.1f}",
            'Wins': self.total_wins,
            'Eps': f"{train_metrics.get('Performance/epsilon', 0):.2f}"
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
        game.display.initialize(game.state.data)
        
        # Run game
        while not game.gameOver:
            for i, agent_obj in enumerate(game.agents):
                if game.gameOver:
                    break
                action = agent_obj.getAction(game.state)
                game.state = game.state.generateSuccessor(i, action)
                game.display.update(game.state.data)
                game.rules.process(game.state, game)
        
        self.agent.epsilon = old_epsilon
        
        return game.state.getScore(), game.state.isWin()
    
    def get_final_summary(self):
        """Get final training summary."""
        recent_scores = self.all_scores[-100:] if len(self.all_scores) >= 100 else self.all_scores
        avg_score = np.mean(recent_scores) if recent_scores else 0
        
        return {
            'Total Wins': f"{self.total_wins}",
            'Avg Score (last 100)': f"{avg_score:.1f}",
            'Final Epsilon': f"{self.agent.epsilon:.3f}"
        }
    
    def get_additional_checkpoint_data(self):
        """Save Q-learning weights in checkpoint."""
        return {
            'qlearning_weights': self.agent.weights,
            'epsilon': self.agent.epsilon,
            'total_wins': self.total_wins
        }


def main():
    parser = argparse.ArgumentParser(description='Q-Learning Pacman Training')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--layout', type=str, default='mediumClassic', help='Layout name')
    parser.add_argument('--render_every', type=int, default=50,
                       help='Render validation game every N episodes (0 to disable)')
    parser.add_argument('--view_speed', type=float, default=0.05, help='Speed of rendered game')
    parser.add_argument('--validation_games', type=int, default=10,
                       help='Number of games to play for validation')
    
    # Q-learning hyperparameters
    parser.add_argument('--alpha', type=float, default=0.2, help='Learning Rate')
    parser.add_argument('--gamma', type=float, default=0.8, help='Discount Factor')
    parser.add_argument('--epsilon', type=float, default=0.05, help='Exploration Rate')
    
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    trainer = QLearningTrainer(
        num_episodes=args.episodes,
        layout_name=args.layout,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        render_every=args.render_every,
        view_speed=args.view_speed,
        validation_games=args.validation_games,
        resume_from=args.resume
    )
    
    trainer.setup()
    trainer.train()


if __name__ == '__main__':
    main()
