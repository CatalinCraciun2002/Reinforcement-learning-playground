"""
CustomPacmanScenario — mediumClassic with randomised capsule positions
and a DirectionalGhost placed near Pacman's corridor.

build_layout()  — loads the layout and randomises the two capsule positions
                  before the game is created.
build_ghosts()  — replaces the second layout ghost slot with a dynamic position
                  near Pacman's spawn, then returns a RandomGhost and a
                  DirectionalGhost for those two slots.
"""

import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agents import ghostAgents
from scenarios.base_scenario import Scenario


class CustomPacmanScenario(Scenario):
    """mediumClassic with randomised capsules and a DirectionalGhost near Pacman."""

    def define_scores(self) -> dict:
        return {
            'food':    10,
            'capsule': 50,
            'ghost':   100,
            'win':     1000,
            'death':  -1000,
            'time':   -1,
        }

    @property
    def layout_name(self) -> str:
        return 'mediumClassic'

    def build_layout(self):
        """Load mediumClassic and randomise the capsule positions before game creation.

        The layout has two capsules — one on the left side and one on the right.
        Each is randomly placed on either the top or bottom row of that side.

        Left  side: x=1,  y ∈ {top_row, bottom_row}
        Right side: x=18, y ∈ {top_row, bottom_row}
        """
        lay = super().build_layout()

        top_row    = lay.height - 2   # first non-wall row from the top
        bottom_row = 1                # first non-wall row from the bottom

        left_y  = random.choice([top_row, bottom_row])
        right_y = random.choice([top_row, bottom_row])

        lay.capsules = [(1, left_y), (lay.width - 2, right_y)]

        return lay

    def build_ghosts(self, lay) -> list:
        """One RandomGhost at layout slot 1, one DirectionalGhost near Pacman's spawn.

        The mediumClassic layout has two G slots. We keep the first (RandomGhost)
        and replace the second slot's position with a dynamic position on a random
        side of Pacman's spawn corridor.
        """
        pacman_pos = next(pos for is_pac, pos in lay.agentPositions if is_pac)
        side = random.choice([-1, 1])
        ghost_pos = (int(pacman_pos[0] + side * 3), int(pacman_pos[1]))

        # Replace the 2nd ghost slot position so the engine places the DirectionalGhost
        # at the dynamic position rather than the fixed G coordinate from the layout
        ghost_slots = [i for i, (is_pac, _) in enumerate(lay.agentPositions) if not is_pac]
        lay.agentPositions[ghost_slots[1]] = (False, ghost_pos)

        return [
            ghostAgents.RandomGhost(1),
            ghostAgents.DirectionalGhost(2, prob_attack=1.0, prob_scaredFlee=0.0),
        ]

    def setup(self, env) -> None:
        pass
