"""
Base Game class — defines the contract for all Pacman game configurations.

Every concrete Game must:
  1. Define define_scores() as its FIRST method
  2. Specify a layout_name
  3. Optionally override add_ghosts / add_power_ups / add_points
     to replace what the layout provides.
"""

from abc import ABC, abstractmethod


class Game(ABC):
    """Abstract base class for a Pacman game configuration."""

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

    # ------------------------------------------------------------------ #
    #  OPTIONAL OVERRIDES                                                  #
    # ------------------------------------------------------------------ #

    def add_ghosts(self, env):
        """Override ghost setup after environment reset.

        When implemented, replaces the layout's default ghosts entirely.
        Receives the PacmanEnv instance so you can call env.add_directional_ghost().
        Leave unimplemented to keep layout defaults.
        """

    def add_power_ups(self, env):
        """Override capsule positions after environment reset.

        When implemented, clears layout capsules and places new ones.
        Receives the PacmanEnv instance.
        Leave unimplemented to keep layout defaults.
        """

    def add_points(self, env):
        """Override food grid after environment reset.

        When implemented, clears layout food and places new pellets.
        Receives the PacmanEnv instance.
        Leave unimplemented to keep layout defaults.
        """

    # ------------------------------------------------------------------ #
    #  SETTINGS                                                            #
    # ------------------------------------------------------------------ #

    @property
    def allow_stop(self) -> bool:
        """Whether Pacman can use the STOP action. Default: True."""
        return True

    @property
    def num_ghosts(self) -> int:
        """Number of ghosts to spawn from layout. Default: layout default."""
        return None  # None = use layout's ghost count

    # ------------------------------------------------------------------ #
    #  INTERNAL: computed once on first access                            #
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

    # ------------------------------------------------------------------ #
    #  HELPERS: detect which overrides are active                         #
    # ------------------------------------------------------------------ #

    def _overrides_ghosts(self) -> bool:
        return type(self).add_ghosts is not Game.add_ghosts

    def _overrides_power_ups(self) -> bool:
        return type(self).add_power_ups is not Game.add_power_ups

    def _overrides_points(self) -> bool:
        return type(self).add_points is not Game.add_points
