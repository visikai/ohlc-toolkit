"""One guard against writing over a column a frame already carries.

Overwriting is never the intent and would be undetectable after the fact:
the replaced column keeps its name, its dtype and its plausibility,
having lost whatever the caller put there.

This lives here, beside the error taxonomy and the echo helper, for the
reason those do: every subpackage that writes a column needs it, and a
guard that lived in one of them would either be imported upward or copied
again. It was copied twice before this module existed --
:mod:`ohlc_toolkit.returns.primitives` and
:mod:`ohlc_toolkit.windows.annotations` each had their own -- and the two
had already diverged, one echoing caller-supplied names unbounded and the
other bounding them.

What each caller keeps is its own REMEDY: the sentence telling a reader
what to do about the collision differs by call site, and only that
sentence does.
"""

from collections.abc import Sequence

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.temporal.echo import bounded_echo
from ohlc_toolkit.temporal.errors import ConfigError

logger = get_logger(__name__)


def require_absent_columns(
    frame: pl.DataFrame, names: Sequence[str], *, remedy: str
) -> None:
    """Refuse to write over a column the frame already carries.

    Args:
        frame: The frame about to be written to.
        names: The column names this call would add.
        remedy: What the caller should do instead, in the calling
            module's own words. It completes the refusal, so write it as
            a sentence.

    Raises:
        ConfigError: If ``frame`` already carries any of ``names``.

    """
    present = [name for name in names if name in frame.columns]
    if not present:
        return
    # Bounded, because a column name comes from a caller and can be as
    # long as a caller likes. One of the two implementations this replaces
    # echoed them raw.
    echoed = ", ".join(bounded_echo(name) for name in present)
    logger.warning("Refusing to overwrite existing column(s): {}", echoed)
    raise ConfigError(f"The frame already carries column(s) {echoed}; {remedy}")
