"""
Base Scenario class — defines the contract for all Pacman game configurations.

Every concrete Scenario must:
  1. Define define_scores() as its FIRST method
  2. Specify a layout_name
  3. Optionally override build_ghosts / add_ghosts / add_power_ups / add_points
     to customise the game after each reset.
"""

from abc import ABC, abstractmethod

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

    # ------------------------------------------------------------------ #
    #  GHOST BUILDING                                                      #
    # ------------------------------------------------------------------ #

    def build_ghosts(self, num_ghosts: int) -> list:
        """Build and return the ghost agent list to pass to ClassicGameRules.newGame().

        Override to fully control which ghosts are created. The default is all
        RandomGhosts. Use add_ghosts() when ghost placement depends on game state
        (e.g. relative to Pacman's spawn position).

        Args:
            num_ghosts: Number of ghosts as determined by the layout (or
                        overridden by self.num_ghosts).
        """
        return [ghostAgents.RandomGhost(i + 1) for i in range(num_ghosts)]

    # ------------------------------------------------------------------ #
    #  POST-RESET SETUP                                                    #
    # ------------------------------------------------------------------ #

    def setup(self, env) -> None:
        """Called once after every env reset to apply scenario-specific overrides.

        The environment calls this instead of managing each override individually.
        Subclasses should override add_ghosts / add_power_ups / add_points — not
        this method — unless they need to coordinate across all three.
        """
        if type(self).add_ghosts is not Scenario.add_ghosts:
            self.add_ghosts(env)
        if type(self).add_power_ups is not Scenario.add_power_ups:
            self.add_power_ups(env)
        if type(self).add_points is not Scenario.add_points:
            self.add_points(env)

    def add_ghosts(self, env) -> None:
        """Inject extra ghost agents after the game is created.

        Override to add ghosts on top of (or instead of) what build_ghosts()
        provides. Receives the PacmanEnv instance.
        Leave unimplemented to keep the ghosts from build_ghosts().
        """

    def add_power_ups(self, env) -> None:
        """Override capsule positions after environment reset.

        When implemented, clears layout capsules and places new ones.
        Leave unimplemented to keep layout defaults.
        """

    def add_points(self, env) -> None:
        """Override food grid after environment reset.

        When implemented, clears layout food and places new pellets.
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
