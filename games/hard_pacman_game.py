"""
HardPacmanGame — mediumClassic with an extra directional ghost near Pacman's spawn.

Inherits all rewards from PacmanGame. Overrides add_ghosts() to inject
a near-spawn DirectionalGhost on top of the layout's default ghosts.
"""

import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.pacman_game import PacmanGame


class HardPacmanGame(PacmanGame):
    """mediumClassic + extra directional ghost near Pacman spawn."""

    # ------------------------------------------------------------------ #
    #  REWARDS  — first method, mandatory                                 #
    # ------------------------------------------------------------------ #

    def define_scores(self) -> dict:
        # Same rewards as PacmanGame — override here to adjust difficulty
        return super().define_scores()

    # ------------------------------------------------------------------ #
    #  GHOST OVERRIDE                                                      #
    # ------------------------------------------------------------------ #

    def add_ghosts(self, env):
        """Add an extra DirectionalGhost on a random side of Pacman's spawn."""
        pacman_pos = env.game.state.getPacmanPosition()
        side = random.choice([-1, 1])
        ghost_x = int(pacman_pos[0] + side * 3)
        ghost_y = int(pacman_pos[1])
        env.add_directional_ghost((ghost_x, ghost_y), prob_attack=1.0, prob_scaredFlee=0.0)
