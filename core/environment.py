"""
Pacman RL Environment Wrapper

PacmanEnv is a thin game runner. All configuration — layout, ghosts, rewards,
post-reset hooks — lives in the Scenario object passed at construction time.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pacman import ClassicGameRules
from display import textDisplay
from core.game import Directions


class PacmanEnv:
    """Pacman environment driven by a Scenario configuration object."""

    def __init__(self, agent, scenario, display=None, env_id=0):
        """
        Args:
            agent:    The Pacman RL agent.
            scenario: A Scenario instance defining layout, ghosts, and rewards.
            display:  Optional display (default: headless NullGraphics).
            env_id:   Integer ID used for multi-environment memory contexts.
        """
        self.agent = agent
        self._scenario = scenario
        self._game = None         # Running core.Game instance
        self.wins = 0
        self.display = display or textDisplay.NullGraphics()
        self.env_id = env_id
        self.epoch = 0            # Set by the training loop; synced to scenario each step
        self._score = 0.0         # Accumulated raw score for the current episode
        self.reset()

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                          #
    # ------------------------------------------------------------------ #

    @property
    def scenario(self):
        """The active Scenario configuration object."""
        return self._scenario

    @property
    def game(self):
        """The running core.Game instance (holds .state, .agents, etc.)."""
        return self._game

    @property
    def state(self):
        """Shortcut to the current game state."""
        return self._game.state

    def set_scenario(self, scenario):
        """Swap the active scenario and reset the environment.

        Useful for curriculum training — change layout or rules mid-training
        without recreating the env or agent.

        Args:
            scenario: New Scenario instance to use from the next episode onward.
        """
        self._scenario = scenario
        self.reset()

    def reset(self):
        """Reset to a fresh episode and return the initial game state."""
        lay = self._scenario.build_layout()
        ghosts = self._scenario.build_ghosts(lay)

        rules = ClassicGameRules()
        self._game = rules.newGame(
            lay, self.agent, ghosts, self.display,
            quiet=True, catchExceptions=False
        )
        self.agent.registerInitialState(self._game.state, self.env_id)
        self.display.initialize(self._game.state.data)

        self._score = 0.0
        self._scenario.setup(self)

        return self._game.state

    def step(self, action):
        """Execute one action and return (next_state, scaled_reward, done)."""
        gi = self._game

        # Sync epoch so scenario can read self.epoch in any hook
        self._scenario.epoch = self.epoch

        prev_food      = gi.state.getNumFood()
        prev_capsules  = len(gi.state.getCapsules())
        prev_scared    = [g.scaredTimer for g in gi.state.getGhostStates()]

        # Pacman moves
        gi.state = gi.state.generateSuccessor(0, action)
        gi.display.update(gi.state.data)

        # Ghosts move
        for i in range(1, len(gi.agents)):
            if gi.state.isWin() or gi.state.isLose():
                break
            ghost_action = gi.agents[i].getAction(gi.state)
            gi.state = gi.state.generateSuccessor(i, ghost_action)
            gi.display.update(gi.state.data)

        gi.rules.process(gi.state, gi)

        # Reward accumulation
        reward = self._scenario.raw('time')

        food_eaten     = prev_food - gi.state.getNumFood()
        reward        += food_eaten * self._scenario.raw('food')

        capsules_eaten = prev_capsules - len(gi.state.getCapsules())
        reward        += capsules_eaten * self._scenario.raw('capsule')

        curr_scared = [g.scaredTimer for g in gi.state.getGhostStates()]
        for psc, csc in zip(prev_scared, curr_scared):
            if psc > 0 and csc == 0:
                reward += self._scenario.raw('ghost')

        done = gi.gameOver
        if done:
            if gi.state.isWin():
                reward += self._scenario.raw('win')
                self.wins += 1
            elif gi.state.isLose():
                reward += self._scenario.raw('death')

        # Keep UI score in sync with our scenario's scoring
        self._score += reward
        gi.state.data.score = int(self._score)

        return gi.state, reward / self._scenario.reward_scale(), done

    @property
    def is_over(self):
        return self._game.gameOver

    def get_legal(self, state):
        legal = state.getLegalPacmanActions()
        if not self._scenario.allow_stop and Directions.STOP in legal:
            legal.remove(Directions.STOP)
        return legal
