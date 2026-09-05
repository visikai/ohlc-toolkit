"""One bounded way to echo an untrusted value into a log or message.

Rejected input is quoted back to the caller so a refusal names what it
refused -- but echoing is how a pathological input turns one log line
into a megabyte. Every module that echoes therefore uses this one
helper: ``repr`` so a string shows its quotes and escape sequences
rather than smuggling control characters into a terminal, one cap with
one stated size, and a length note so the reader of a truncated echo
still learns how large the original was.

Reading an enum member out of a payload lives here too, because the only
reason that refusal is interesting is the echo it has to make: the value
it rejects came from outside and is the caller's to size.

The rule, package-wide: a value this package did not choose -- file
content, a payload field, a dtype, a name, a tag, a path, a URL, the text
of a third-party error -- reaches a log line or an error message only
through :func:`bounded_echo`. A refusal that is about the value's TYPE
logs ``type(value).__name__`` instead, matching what its exception says,
because the type name is the whole diagnostic and the value adds nothing
a bound would then have to trim. Three kinds of value need no bound, and
they are the only ones echoed without one: first-party literals, keys
and counts; numbers a guard has already validated; and values
bounded where they were made -- a manifest asset name the parser caps in
length before a record can exist, or a column name this package composes
from an enum member and a canonical duration label. The rule is enforced
at each site by that site's own test, never by a truncating backstop in
the log sink: a sink that trimmed silently would hide exactly the defects
those tests exist to catch, and would mangle legitimately long structured
output while doing it. The sink therefore bounds nothing itself; the text
of an exception this package raises is bounded by the site that raised
it, and a third-party exception crossing the sink carries whatever text
its author gave it.
"""

from enum import Enum
from typing import TypeVar

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.temporal.errors import ConfigError

logger = get_logger(__name__)

_EnumMember = TypeVar("_EnumMember", bound=Enum)

# Eighty characters names any ordinary dtype, tag, column, or duration
# string while keeping a single log line readable.
MAX_ECHO_CHARS = 80


def bounded_echo(value: object) -> str:
    """Render ``value`` for a log line or error message, bounded.

    Args:
        value: Any value, typically one read straight from untrusted
            input.

    Returns:
        ``repr(value)``, truncated with a note stating the full
        representation length once it exceeds :data:`MAX_ECHO_CHARS`.

    """
    text = repr(value)
    if len(text) <= MAX_ECHO_CHARS:
        return text
    return f"{text[:MAX_ECHO_CHARS]}... ({len(text)} chars total)"


def enum_from_payload(
    enum_type: type[_EnumMember], value: object, *, label: str
) -> _EnumMember:
    """Read an enum member out of a payload by its stored value.

    Args:
        enum_type: The enum the stored value must name a member of.
        value: The stored value, of any type.
        label: What the member is, for the message.

    Returns:
        The named member.

    Raises:
        ConfigError: If ``value`` names no member. The message lists the
            members that do exist, and echoes the offending value
            through :func:`~ohlc_toolkit.temporal.bounded_echo` so an
            oversized string cannot produce an oversized message.

    """
    try:
        return enum_type(value)
    except ValueError as error:
        quoted = bounded_echo(value)
        logger.warning("Rejecting an unknown {}: {}", label, quoted)
        raise ConfigError(
            f"Unknown {label}: {quoted}. Supported: "
            f"{[member.value for member in enum_type]}."
        ) from error
