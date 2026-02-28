"""
ContoursMazeScenario — Pacman on the contoursMaze layout.

Maze-heavy layout with winding corridors — tests navigation in tight spaces.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios.base_scenario import Scenario


class ContoursMazeScenario(Scenario):
    """Pacman game on the contoursMaze layout."""

    def define_scores(self) -> dict:
        return {
            'food':    10,
            'capsule': 50,
            'ghost':   200,
            'win':     1000,
            'death':  -1000,
            'time':   -1,
        }

    @property
    def layout_name(self) -> str:
        return 'contoursMaze'
