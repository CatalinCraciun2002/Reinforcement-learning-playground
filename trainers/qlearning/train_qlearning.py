"""
Approximate Q-Learning Training Script - Refactored with BaseTrainer
"""
import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from reinforcement_learning.base_trainer import BaseTrainer
from agents.qlearning_agents.qlearning_agent import ApproximateQAgent
from core.pacman import ClassicGameRules
import display.textDisplay as textDisplay
import display.graphicsDisplay as graphicsDisplay
from core.game import Directions
from scenarios.scenario_suites import SUITES


class QLearningTrainer(BaseTrainer):
    """Q-Learning Trainer using BaseTrainer framework."""

    def __init__(
        self,
        num_episodes=100,
        suite_name='medium_classic_only',
        alpha=0.2,
        gamma=0.8,
        epsilon=0.05,
        render_every=0,
        view_speed=0.05,
        validation_games=10,
        resume_from=None
    ):
        self.suite_name = suite_name
        self.alpha = alpha
        self.gamma = gamma
        self.initial_epsilon = epsilon
        self.render_every = render_every
        self.view_speed = view_speed
        self.validation_games = validation_games

        hyperparams = {
            'num_episodes': num_episodes,
            'suite': suite_name,
            'alpha': alpha,
            'gamma': gamma,
            'epsilon': epsilon,
            'validation_games': validation_games
        }

        super().__init__(
            training_type='qlearning',
            num_epochs=num_episodes,
            hyperparams=hyperparams,
            resume_from=resume_from,
            use_best_checkpoint=False
        )

        self.agent = None
        self._suite = None
        self._rules = ClassicGameRules()
        self.total_wins = 0
        self.all_scores = []

    def create_model(self):
        """Q-learning uses feature weights, not a neural network — return a dummy module."""
        class DummyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))
        return DummyModule()

    def create_optimizer(self, model):
        """Q-learning doesn't use an optimizer."""
        return None

    def post_setup(self):
        """Setup Q-learning agent and scenario suite."""
        self.agent = ApproximateQAgent(
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=self.initial_epsilon,
            numTraining=self.num_epochs
        )

        if self.logger.checkpoint_data and 'qlearning_weights' in self.logger.checkpoint_data:
            self.agent.weights = self.logger.checkpoint_data['qlearning_weights']
            print(f"  ✓ Q-learning weights loaded from checkpoint")

        self._suite = SUITES[self.suite_name]

    def _pick_scenario(self):
        """Pick a scenario from the training suite."""
        import random
        if self._suite.mode == 'successive':
            idx = getattr(self, '_suite_cursor', 0)
            scenario = self._suite.scenarios[idx % len(self._suite.scenarios)][0]
            self._suite_cursor = idx + 1
        else:
            weights = [w for _, w in self._suite.scenarios]
            scenario = random.choices(self._suite.scenarios, weights=weights, k=1)[0][0]
        return scenario

    def _run_game(self, display=None):
        """Run a single game episode. Returns the completed game object."""
        scenario = self._pick_scenario()
        lay = scenario.build_layout()
        ghosts = scenario.build_ghosts(lay)

        game = self._rules.newGame(
            lay, self.agent, ghosts,
            display or textDisplay.NullGraphics(),
            quiet=True, catchExceptions=False
        )
        game.display.initialize(game.state.data)
        return game

    def train_epoch(self, epoch):
        """Train for one episode."""
        game = self._run_game()

        while not game.gameOver:
            for i, agent_obj in enumerate(game.agents):
                if game.gameOver:
                    break

                state = game.state
                action = agent_obj.getAction(state)
                game.state = game.state.generateSuccessor(i, action)
                game.display.update(game.state.data)
                game.rules.process(game.state, game)

                if i == 0:
                    next_state = game.state
                    reward = next_state.getScore() - state.getScore()

                    if game.gameOver:
                        if game.state.isWin():
                            reward += 500
                            self.total_wins += 1
                        elif game.state.isLose():
                            reward -= 500

                    self.agent.update(state, action, next_state, reward)

        final_score = game.state.getScore()
        self.all_scores.append(final_score)

        if self.agent.epsilon > 0.05:
            self.agent.epsilon *= 0.99

        return {
            'Performance/epsilon': self.agent.epsilon,
            'Performance/score': final_score
        }

    def validate(self, epoch):
        """Run validation games."""
        val_scores = []
        val_wins = 0

        train_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.05

        for _ in range(self.validation_games):
            game = self._run_game()

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

        self.agent.epsilon = train_epsilon

        return {
            'Score/validation': np.mean(val_scores),
            'Score/win_rate': val_wins / self.validation_games
        }

    def get_metric_for_checkpoint(self, val_metrics):
        """Use validation score as the metric for checkpointing."""
        return val_metrics['Score/validation'], 'val_score'

    def get_progress_bar_dict(self, train_metrics, val_metrics):
        """Customize progress bar display."""
        recent = self.all_scores[-100:] if self.all_scores else [0]
        return {
            'AvgScore': f"{np.mean(recent):.1f}",
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

            train_epsilon = self.agent.epsilon
            self.agent.epsilon = 0.05

            display = graphicsDisplay.PacmanGraphics(1.0, frameTime=self.view_speed)
            game = self._run_game(display=display)

            while not game.gameOver:
                for i, agent_obj in enumerate(game.agents):
                    if game.gameOver:
                        break
                    action = agent_obj.getAction(game.state)
                    game.state = game.state.generateSuccessor(i, action)
                    game.display.update(game.state.data)
                    game.rules.process(game.state, game)

            self.agent.epsilon = train_epsilon
            score, won = game.state.getScore(), game.state.isWin()
            pbar.write(f"Validation Result: Score={score}, {'WON!' if won else 'Lost'}")
            pbar.write('='*60 + '\n')

    def get_final_summary(self):
        """Get final training summary."""
        recent = self.all_scores[-100:] if self.all_scores else [0]
        return {
            'Total Wins': str(self.total_wins),
            'Avg Score (last 100)': f"{np.mean(recent):.1f}",
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
    parser.add_argument('--suite', type=str, default='medium_classic_only',
                        help=f'Scenario suite name. Available: {list(SUITES.keys())}')
    parser.add_argument('--render-every', type=int, default=10,
                        help='Render validation game every N episodes (0 to disable)')
    parser.add_argument('--view-speed', type=float, default=0.05, help='Speed of rendered game')
    parser.add_argument('--validation-games', type=int, default=10,
                        help='Number of games to play for validation')
    parser.add_argument('--alpha', type=float, default=0.2, help='Learning Rate')
    parser.add_argument('--gamma', type=float, default=0.8, help='Discount Factor')
    parser.add_argument('--epsilon', type=float, default=0.05, help='Exploration Rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    trainer = QLearningTrainer(
        num_episodes=args.episodes,
        suite_name=args.suite,
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
