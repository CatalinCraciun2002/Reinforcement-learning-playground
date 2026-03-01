"""
Base Scenario class — defines the contract for all Pacman game configurations.

Every concrete Scenario must:
  1. Define define_scores() as its FIRST method
  2. Specify a layout_name property
  3. Optionally override build_ghosts(lay) to customise ghost placement.
     The default builds one RandomGhost per ghost position in the layout.
  4. Optionally override add_power_ups(env) / add_points(env) for post-reset
     food/capsule overrides that need the live game state.
"""

from abc import ABC, abstractmethod

from core import layout as layout_module
from agents.base_agents import ghostAgents


class Scenario(ABC):
    """Abstract base class for a Pacman game configuration."""

    def __init__(self):
        self.epoch: int = 0  # Updated by PacmanEnv.step() each training step

    # ------------------------------------------------------------------ #
    #  REWARDS  (must be first method in every subclass)                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def define_scores(self) -> dict:
        """Return raw reward values for each game event.

        Required keys:
            'food'    – reward per food pellet eaten
            'capsule' – reward for eating a power capsule
            'ghost'   – reward for eating a scared ghost
            'win'     – terminal reward for clearing all food
            'death'   – terminal penalty for dying (use negative)
            'time'    – per-step penalty (use negative to discourage stalling)

        All values are automatically scaled to the [-1, 1] range by
        dividing by max(abs(values)).  You only need the raw numbers.
        """

    # ------------------------------------------------------------------ #
    #  LAYOUT                                                              #
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def layout_name(self) -> str:
        """Name of the .lay file (without extension) to load."""

    def build_layout(self):
        """Load and return the Layout object for this scenario.

        Override to return a custom/procedurally generated Layout.
        The default loads the layout file specified by layout_name.
        """
        return layout_module.getLayout(self.layout_name)

    # ------------------------------------------------------------------ #
    #  GHOST BUILDING                                                      #
    # ------------------------------------------------------------------ #

    def build_ghosts(self, lay) -> list:
        """Build and return the ghost agent list for this scenario.

        Receives the Layout object so it can inspect ghost positions,
        Pacman's spawn position, walls, etc. without needing the live
        game state.

        The default creates one RandomGhost per ghost slot in the layout.
        Override to fully control ghost types and placement.

        Args:
            lay: The Layout object returned by build_layout().
        """
        return [ghostAgents.RandomGhost(i + 1) for i in range(lay.getNumGhosts())]

    # ------------------------------------------------------------------ #
    #  POST-RESET SETUP                                                    #
    # ------------------------------------------------------------------ #

    def setup(self, env) -> None:
        """Called once after every env reset. Override for post-reset changes
        that require the live game state (e.g. dynamic actor injection).
        """


    # ------------------------------------------------------------------ #
    #  SETTINGS                                                            #
    # ------------------------------------------------------------------ #

    @property
    def allow_stop(self) -> bool:
        """Whether Pacman can use the STOP action. Default: True."""
        return True

    # ------------------------------------------------------------------ #
    #  REWARD HELPERS  (internal, computed once on first access)          #
    # ------------------------------------------------------------------ #

    _scores_cache = None
    _scale_cache = None

    def _get_scores(self) -> dict:
        if self._scores_cache is None:
            self._scores_cache = self.define_scores()
        return self._scores_cache

    def reward_scale(self) -> float:
        """The divisor used to normalise all rewards to [-1, 1]."""
        if self._scale_cache is None:
            values = self._get_scores().values()
            self._scale_cache = max(abs(v) for v in values)
        return self._scale_cache

    def scaled(self, key: str) -> float:
        """Return the scaled reward for the given event key."""
        return self._get_scores()[key] / self.reward_scale()

    def raw(self, key: str) -> float:
        """Return the raw (unscaled) reward for the given event key."""
        return self._get_scores()[key]
