"""
StandardPacmanScenario — standard mediumClassic configuration.

Baseline scenario: medium layout, one RandomGhost + one DirectionalGhost.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agents import ghostAgents
from scenarios.base_scenario import Scenario


class MediumClassicPacmanScenario(Scenario):
    """Standard Pacman game on mediumClassic layout."""

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
        return 'mediumClassic'

    def build_ghosts(self, num_ghosts: int) -> list:
        """One RandomGhost and one DirectionalGhost."""
        return [
            ghostAgents.RandomGhost(1),
            ghostAgents.DirectionalGhost(2, prob_attack=0.8, prob_scaredFlee=0.8),
        ]
