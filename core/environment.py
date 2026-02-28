"""
Pacman RL Environment Wrapper

PacmanEnv accepts a Game instance (from games/) instead of a layout name.
The Game defines the layout, reward values, and optional overrides for
ghosts, food, and capsules.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import layout
from core.pacman import ClassicGameRules
from display import textDisplay
from agents.base_agents import ghostAgents
from core.game import Directions, AgentState, Configuration
from agents.base_agents.ghostAgents import DirectionalGhost


class PacmanEnv:
    """Pacman environment driven by a Game configuration object."""

    def __init__(self, agent, game, display=None, env_id=0):
        """
        Args:
            agent:       The Pacman RL agent.
            game:        A Game instance (from games/) defining layout + rewards + overrides.
            display:     Optional display (default: headless NullGraphics).
            env_id:      Integer ID used for multi-environment memory contexts.
        """
        self.agent = agent
        self.game_config = game   # Game config (layout, rewards, overrides)
        self._game = None         # Running core.Game instance
        self.wins = 0
        self.display = display or textDisplay.NullGraphics()
        self.env_id = env_id
        self.reset()

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                          #
    # ------------------------------------------------------------------ #

    @property
    def game(self):
        """The running core.Game instance (holds .state, .agents, etc.)."""
        return self._game

    @property
    def state(self):
        """Shortcut to the current game state."""
        return self._game.state

    def set_game(self, game):
        """Swap the active game configuration and reset the environment.

        Useful for curriculum training — change layout or rules mid-training
        without recreating the env or agent.

        Args:
            game: New Game instance to use from the next episode onward.
        """
        self.game_config = game
        self.reset()

    def reset(self):
        """Reset to a fresh episode and return the initial game state."""
        lay = layout.getLayout(self.game_config.layout_name)

        # Build ghost list — use layout count unless game specifies otherwise
        num_ghosts = self.game_config.num_ghosts or lay.getNumGhosts()
        ghosts = (
            [ghostAgents.DirectionalGhost(1, 0.8, 0.8)]
            + [ghostAgents.RandomGhost(i + 1) for i in range(1, num_ghosts)]
        )

        rules = ClassicGameRules()
        self._game = rules.newGame(
            lay, self.agent, ghosts, self.display,
            quiet=True, catchExceptions=False
        )
        self.agent.registerInitialState(self._game.state, self.env_id)
        self.display.initialize(self._game.state.data)

        # Apply optional overrides (only if the Game subclass defines them)
        if self.game_config._overrides_ghosts():
            self.game_config.add_ghosts(self)
        if self.game_config._overrides_power_ups():
            self.game_config.add_power_ups(self)
        if self.game_config._overrides_points():
            self.game_config.add_points(self)

        return self._game.state

    def step(self, action):
        """Execute one action and return (next_state, scaled_reward, done)."""
        gi = self._game

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
        reward = self.game_config.raw('time')

        food_eaten     = prev_food - gi.state.getNumFood()
        reward        += food_eaten * self.game_config.raw('food')

        capsules_eaten = prev_capsules - len(gi.state.getCapsules())
        reward        += capsules_eaten * self.game_config.raw('capsule')

        curr_scared = [g.scaredTimer for g in gi.state.getGhostStates()]
        for psc, csc in zip(prev_scared, curr_scared):
            if psc > 0 and csc == 0:
                reward += self.game_config.raw('ghost')

        done = gi.gameOver
        if done:
            if gi.state.isWin():
                reward += self.game_config.raw('win')
                self.wins += 1
            elif gi.state.isLose():
                reward += self.game_config.raw('death')

        return gi.state, reward / self.game_config.reward_scale(), done

    @property
    def is_over(self):
        return self._game.gameOver

    def get_legal(self, state):
        legal = state.getLegalPacmanActions()
        if not self.game_config.allow_stop and Directions.STOP in legal:
            legal.remove(Directions.STOP)
        return legal

    # ------------------------------------------------------------------ #
    #  LOW-LEVEL HELPERS (used by Game subclasses)                        #
    # ------------------------------------------------------------------ #

    def add_directional_ghost(self, position, prob_attack=0.8, prob_scaredFlee=0.8):
        """Add a DirectionalGhost at the given (x, y) position.

        Called from Game.add_ghosts() implementations.
        """
        gi = self._game
        ghost_index = len(gi.agents)
        ghost_agent = DirectionalGhost(ghost_index, prob_attack, prob_scaredFlee)
        gi.agents.append(ghost_agent)

        ghost_state = AgentState(Configuration(position, Directions.STOP), False)
        gi.state.data.agentStates.append(ghost_state)

        if hasattr(gi.display, 'agentImages'):
            img = gi.display.drawGhost(ghost_state, ghost_index)
            gi.display.agentImages.append((ghost_state, img))
