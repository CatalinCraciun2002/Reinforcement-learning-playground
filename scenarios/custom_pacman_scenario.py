"""
HardPacmanScenario — mediumClassic with a single DirectionalGhost spawned on a
random side of Pacman's corridor after the game is created.

build_ghosts() returns [] so no layout-default ghosts are added.
add_ghosts() (called via setup()) places one DirectionalGhost near Pacman once
the game state is available.
"""

import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agents import ghostAgents
from core.game import AgentState, Configuration, Directions
from scenarios.base_scenario import Scenario


class CustomPacmanScenario(Scenario):
    """mediumClassic with one DirectionalGhost placed near Pacman's spawn."""

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
        """No layout-based ghosts — the ghost is placed manually in add_ghosts()."""
        return [
            ghostAgents.RandomGhost(1),
        ]

    def add_ghosts(self, env):
        """Spawn one DirectionalGhost on a random side of Pacman's spawn position."""
        gi = env.game
        pacman_pos = gi.state.getPacmanPosition()
        side = random.choice([-1, 1])
        position = (int(pacman_pos[0] + side * 3), int(pacman_pos[1]))

        ghost_index = len(gi.agents)
        ghost_agent = ghostAgents.DirectionalGhost(ghost_index, prob_attack=1.0, prob_scaredFlee=0.0)
        gi.agents.append(ghost_agent)

        ghost_state = AgentState(Configuration(position, Directions.STOP), False)
        gi.state.data.agentStates.append(ghost_state)

        if hasattr(gi.display, 'agentImages'):
            img = gi.display.drawGhost(ghost_state, ghost_index)
            gi.display.agentImages.append((ghost_state, img))
