"""Public exception taxonomy for the temporal package.

The hierarchy is intentionally flat: every exception here inherits directly
from ``Exception`` rather than from a shared base, so callers can catch each
concern independently without accidentally widening a ``except`` clause.

The log line before a raise follows the same split. A ``ConfigError`` (or
any refusal of the caller's own argument) is preceded by ``logger.warning``:
the caller wrote the value, can read the message, and can fix the call. A
``DataValidationError``, a ``CoverageError``, an integrity failure or a
missing file is preceded by ``logger.error``: something outside the call --
a file, a download, a manifest, a sidecar, a source frame -- turned out not
to be what it claimed, and that is worth an operator's attention even when
the exception is caught. A bare ``raise`` that hands a caught failure on
unchanged sits on the error side too, and the file branch is ``OSError`` in
full, so a permission or connection failure stands beside a missing file. A
``polars.exceptions.PolarsError`` raised or re-raised by this package is on
the error side for the same reason: it says the bytes handed in could not be
read as what they claimed to be.
A new exception class must be classified into one branch or the other before
it is raised anywhere: ``tests/test_refusal_levels.py`` holds every ``raise``
in the package to this pairing and fails a class on neither branch.
"""


class ConfigError(Exception):
    """A duration, cadence, or schedule could not be resolved.

    Raised at resolution time when a configuration value is malformed or
    violates an invariant (for example, a negative duration or a duration
    string outside the supported grammar). This is always an error, never
    merely warned about and continued: callers cannot proceed with an
    unresolved configuration. The log line before it is nevertheless a
    ``warning``, because the caller can fix the call.
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
