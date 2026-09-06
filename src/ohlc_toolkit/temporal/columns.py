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

from collections.abc import Collection

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.temporal.echo import bounded_echo
from ohlc_toolkit.temporal.errors import ConfigError

logger = get_logger(__name__)


def require_absent_columns(
    frame: pl.DataFrame, names: Collection[str], *, remedy: str
) -> None:
    """Refuse to write over a column the frame already carries.

    Args:
        frame: The frame about to be written to.
        names: The column names this call would add. A bare ``str`` is
            REFUSED rather than iterated -- see below.
        remedy: What the caller should do instead, in the calling
            module's own words. It completes the refusal, so write it as
            a sentence.

    Raises:
        ConfigError: If ``names`` is a single string, or if ``frame``
            already carries any of ``names``.

    """
    # A `str` IS a `Collection[str]`, so no annotation can make this
    # unrepresentable and a type checker will not catch it. Passing one
    # iterates its CHARACTERS: `require_absent_columns(frame, "close")`
    # looks for columns named "c", "l", "o", "s", "e", finds none, and
    # returns -- a guard whose whole purpose is to prevent an overwrite
    # passing while the overwrite proceeds. It is the most natural
    # mistake a caller can make and the worst failure this function has,
    # so it is refused out loud rather than left to a checker the caller
    # may not run.
    if isinstance(names, str):
        logger.warning("Rejecting a single column name passed as a bare str.")
        raise ConfigError(
            f"names must be a collection of column names, not a single string; "
            f"{bounded_echo(names)} would be read one character at a time. Pass "
            "a tuple, as in (name,)."
        )
    present = [name for name in names if name in frame.columns]
    if not present:
        return
    # Bounded, because a column name comes from a caller and can be as
    # long as a caller likes. One of the two implementations this replaces
    # echoed them raw.
    echoed = ", ".join(bounded_echo(name) for name in present)
    logger.warning("Refusing to overwrite existing column(s): {}", echoed)
    raise ConfigError(f"The frame already carries column(s) {echoed}; {remedy}")
