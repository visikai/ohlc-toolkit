"""One bounded way to echo an untrusted value into a log or message.

Rejected input is quoted back to the caller so a refusal names what it
refused -- but echoing is how a pathological input turns one log line
into a megabyte. Every module that echoes therefore uses this one
helper: ``repr`` so a string shows its quotes and escape sequences
rather than smuggling control characters into a terminal, one cap with
one stated size, and a length note so the reader of a truncated echo
still learns how large the original was.
"""

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
