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

    Strict source-frame validation raises this, typically via a subclass
    that carries a structured findings report (for example
    ``ohlc_toolkit.source.validation.SourceValidationError``) rather than
    this base class directly, so callers can inspect exactly what failed
    instead of parsing a message.
    """


class CoverageError(Exception):
    """A window failed a strict data-quality or coverage gate.

    A strict window quality gate raises this, typically via a subclass
    that carries a structured quality report (for example
    ``ohlc_toolkit.windows.quality.WindowCoverageError``) rather than
    this base class directly, so callers can inspect exactly which rows
    failed instead of parsing a message.
    """
