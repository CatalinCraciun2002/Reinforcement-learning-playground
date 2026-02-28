"""
Scenario Suites — predefined collections of scenarios for training and testing.

Each suite has a mode:
  - 'successive':    play each scenario in order, cycling through the list
  - 'probabilistic': pick a random scenario weighted by probability
"""

from dataclasses import dataclass
from typing import List, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios.medium_classic_pacman_scenario import MediumClassicPacmanScenario
from scenarios.custom_pacman_scenario import CustomPacmanScenario
from scenarios.open_classic_scenario import OpenClassicScenario
from scenarios.contours_maze_scenario import ContoursMazeScenario
from scenarios.base_scenario import Scenario


@dataclass
class ScenarioSuite:
    """A named collection of weighted scenarios with a selection mode."""
    mode: str                                    # 'successive' or 'probabilistic'
    scenarios: List[Tuple[Scenario, float]]      # (scenario_instance, weight)

    def __post_init__(self):
        if self.mode not in ('successive', 'probabilistic'):
            raise ValueError(f"Suite mode must be 'successive' or 'probabilistic', got '{self.mode}'")
        if not self.scenarios:
            raise ValueError("Suite must have at least one scenario")


# ------------------------------------------------------------------ #
#  Registry of available suites                                       #
# ------------------------------------------------------------------ #

SUITES = {
    'custom_only': ScenarioSuite(
        mode='probabilistic',
        scenarios=[(CustomPacmanScenario(), 1.0)],
    ),
    'medium_classic_only': ScenarioSuite(
        mode='probabilistic',
        scenarios=[(MediumClassicPacmanScenario(), 1.0)],
    ),
    'open_classic': ScenarioSuite(
        mode='probabilistic',
        scenarios=[(OpenClassicScenario(), 1.0)],
    ),
    'contours_maze': ScenarioSuite(
        mode='probabilistic',
        scenarios=[(ContoursMazeScenario(), 1.0)],
    ),
    'mixed': ScenarioSuite(
        mode='probabilistic',
        scenarios=[(MediumClassicPacmanScenario(), 0.3), (CustomPacmanScenario(), 0.7)],
    ),
    'all_layouts': ScenarioSuite(
        mode='successive',
        scenarios=[
            (MediumClassicPacmanScenario(), 1.0),
            (CustomPacmanScenario(), 1.0),
            (OpenClassicScenario(), 1.0),
            (ContoursMazeScenario(), 1.0),
        ],
    ),
}
