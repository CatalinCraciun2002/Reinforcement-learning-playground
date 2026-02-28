"""
PacmanGame — standard mediumClassic configuration.

This is the baseline game: medium layout, default ghosts from layout,
no extra rules. Use this as a starting point for custom games.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.base_game import Game


class PacmanGame(Game):
    """Standard Pacman game on mediumClassic layout."""

    # ------------------------------------------------------------------ #
    #  REWARDS  — first method, mandatory                                 #
    # ------------------------------------------------------------------ #

    def define_scores(self) -> dict:
        return {
            'food':    10,
            'capsule': 50,
            'ghost':   200,
            'win':     1000,
            'death':  -1000,
            'time':   -1,
        }

    # ------------------------------------------------------------------ #
    #  LAYOUT                                                              #
    # ------------------------------------------------------------------ #

    @property
    def layout_name(self) -> str:
        return 'mediumClassic'
