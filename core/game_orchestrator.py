"""
GameOrchestrator — manages training and testing environments.

Wraps all PacmanEnv interactions so the trainer never touches envs directly.
Loads scenario suites to control which scenarios are played during training
and testing.
"""

import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.environment import PacmanEnv
from scenarios.scenario_suites import SUITES, ScenarioSuite
from display import graphicsDisplay


class GameOrchestrator:
    """Manages batch training environments and validation using scenario suites."""

    def __init__(self, agent, batch_size, train_suite_name='hard_only', test_suite_name='hard_only'):
        """
        Args:
            agent:            The Pacman RL agent.
            batch_size:       Number of parallel training environments.
            train_suite_name: Name of the suite (from SUITES) for training.
            test_suite_name:  Name of the suite (from SUITES) for testing.
        """
        self.agent = agent
        self._batch_size = batch_size
        self.train_suite_name = train_suite_name
        self.test_suite_name = test_suite_name

        self._train_suite = SUITES[train_suite_name]
        self._test_suite = SUITES[test_suite_name]

        # Per-env successive cursors (index into the suite's scenario list)
        self._train_cursors = [0] * batch_size

        # Training environments
        self._envs = [
            PacmanEnv(agent, self._pick_train_scenario(i), env_id=i)
            for i in range(batch_size)
        ]

    # ------------------------------------------------------------------ #
    #  Scenario Selection                                                  #
    # ------------------------------------------------------------------ #

    def _pick_scenario(self, suite, cursor=None):
        """Pick a scenario from a suite. Returns (scenario, new_cursor).

        For successive mode, cursor tracks position; for probabilistic, cursor is unused.
        """
        if suite.mode == 'successive':
            idx = cursor % len(suite.scenarios)
            scenario = suite.scenarios[idx][0]
            return scenario, cursor + 1
        else:
            weights = [w for _, w in suite.scenarios]
            chosen = random.choices(suite.scenarios, weights=weights, k=1)[0]
            return chosen[0], cursor

    def _pick_train_scenario(self, env_idx):
        """Pick the next training scenario for the given env index."""
        scenario, self._train_cursors[env_idx] = self._pick_scenario(
            self._train_suite, self._train_cursors[env_idx]
        )
        return scenario

    # ------------------------------------------------------------------ #
    #  Per-Environment Methods                                             #
    # ------------------------------------------------------------------ #

    def step(self, env_idx, action):
        """Execute one action on a training env. Returns (state, reward, done)."""
        return self._envs[env_idx].step(action)

    def reset(self, env_idx):
        """Reset a training env with the next scenario from the training suite."""
        scenario = self._pick_train_scenario(env_idx)
        self._envs[env_idx].set_scenario(scenario)

    def get_legal(self, env_idx, state):
        """Get legal actions for a training env."""
        return self._envs[env_idx].get_legal(state)

    def get_state(self, env_idx):
        """Get the current game state of a training env."""
        return self._envs[env_idx].game.state

    def get_wins(self, env_idx):
        """Get the win count of a training env."""
        return self._envs[env_idx].wins

    # ------------------------------------------------------------------ #
    #  Batch Methods (all training envs)                                   #
    # ------------------------------------------------------------------ #

    @property
    def batch_size(self):
        return self._batch_size

    def get_all_states(self):
        """Return a list of current game states for all training envs."""
        return [env.game.state for env in self._envs]

    def get_all_legal(self, states):
        """Return a list of legal actions for all training envs given their states."""
        return [env.get_legal(s) for env, s in zip(self._envs, states)]

    def set_epoch(self, epoch):
        """Sync epoch to all training environments."""
        for env in self._envs:
            env.epoch = epoch

    def total_wins(self):
        """Sum of wins across all training environments."""
        return sum(env.wins for env in self._envs)

    def total_wins_plus_one(self):
        """Sum of (wins + 1) across all training envs (for averaging)."""
        return sum(env.wins + 1 for env in self._envs)

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    def run_validation(self, n_games=8, with_graphics=False, max_steps=1000):
        """Run validation games according to the test suite.

        successive:    n_games per scenario in the suite
        probabilistic: n_games total, each sampled by weight

        Returns:
            List of (score, won, steps) tuples.
        """
        suite = self._test_suite
        results = []

        if suite.mode == 'successive':
            for scenario, _ in suite.scenarios:
                for _ in range(n_games):
                    results.append(self._play_one(scenario, with_graphics, max_steps))
        else:
            cursor = 0
            for _ in range(n_games):
                scenario, cursor = self._pick_scenario(suite, cursor)
                results.append(self._play_one(scenario, with_graphics, max_steps))

        return results

    def run_single_validation(self, with_graphics=True, max_steps=1000):
        """Run a single validation game (for epoch-end display).

        successive:    plays the first scenario in the suite
        probabilistic: samples one by weight

        Returns:
            (score, won, steps)
        """
        suite = self._test_suite
        scenario, _ = self._pick_scenario(suite, 0)
        return self._play_one(scenario, with_graphics, max_steps)

    def _play_one(self, scenario, with_graphics, max_steps):
        """Play a single game with the given scenario. Returns (score, won, steps)."""
        import torch

        display = graphicsDisplay.PacmanGraphics(1.0, frameTime=0.05) if with_graphics else None
        val_env = PacmanEnv(self.agent, scenario, display=display)

        steps = 0
        done = False

        while not done and steps < max_steps:
            state = val_env.game.state
            legal = val_env.get_legal(state)

            if not legal:
                break

            with torch.no_grad():
                probs, _ = self.agent.forward(state)

            action, _ = self.agent.getAction(legal, probs)
            _, _, done = val_env.step(action)
            steps += 1

        score = val_env.game.state.getScore()
        won = val_env.game.state.isWin()
        return score, won, steps
