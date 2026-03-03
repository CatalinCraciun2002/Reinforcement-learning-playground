"""
OpenClassicScenario — Pacman on the openClassic layout.

Open layout with fewer walls — tests open-space navigation strategy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios.base_scenario import Scenario


class OpenClassicScenario(Scenario):
    """Pacman game on the openClassic layout."""

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
        return 'openClassic'
