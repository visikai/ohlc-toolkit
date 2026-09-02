"""Public exception taxonomy for the temporal package.

The hierarchy is intentionally flat: every exception here inherits directly
from ``Exception`` rather than from a shared base, so callers can catch each
concern independently without accidentally widening a ``except`` clause.
"""


class ConfigError(Exception):
    """A duration, cadence, or schedule could not be resolved.

    Raised at resolution time when a configuration value is malformed or
    violates an invariant (for example, a negative duration or a duration
    string outside the supported grammar). This is always an error, never
    a warning: callers cannot proceed with an unresolved configuration.
    """


class DataValidationError(Exception):
    """A source data frame failed strict validation.

    Reserved for the temporal package's data-validation surface. No code
    in this package raises it yet; it is defined now so the public
    exception taxonomy is stable for callers to depend on.
    """


class CoverageError(Exception):
    """A window failed a strict data-quality or coverage gate.

    Reserved for the temporal package's window-quality surface. No code
    in this package raises it yet; it is defined now so the public
    exception taxonomy is stable for callers to depend on.
    """
