"""Canonical serialization and content hashing, shared by every schedule type.

A schedule type here records what it is as a plain, JSON-compatible
payload, and names itself by a content hash over that payload. Both
halves live in this module so the window schedules and the cadence rules
cannot drift apart on what "the same identity" means.

The canonical form
------------------

A payload is hashed as ``json.dumps`` with sorted keys, no whitespace,
ASCII escapes, and no NaN or infinity accepted. Sorted keys are what
make the hash a function of the mapping rather than of the order it was
built in, so a payload rebuilt field by field in a different order still
names the same schedule. Durations are stored as their compact strings
(``"1m"``, ``"2w"``), which are the canonical rendering of a Duration
and parse back to exactly the value they came from.

Reading a payload back
----------------------

Every helper here refuses rather than coerces. A payload is a record of
a decision somebody made; quietly repairing a malformed one would
produce a schedule nobody chose, under an id that no longer describes
it.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.temporal import ConfigError, Duration, bounded_echo

logger = get_logger(__name__)


def canonical_json(payload: Mapping[str, object]) -> str:
    """Render an identity payload as the single JSON text it is hashed as.

    Args:
        payload: The identity payload, holding only JSON scalars, lists,
            and nested mappings of the same.

    Returns:
        The canonical JSON text: keys sorted at every level, no
        whitespace, ASCII-escaped.

    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def content_hash(payload: Mapping[str, object]) -> str:
    """Return the sha256 hex digest of a payload's canonical JSON.

    Args:
        payload: The identity payload to name.

    Returns:
        The 64-character lowercase hex digest.

    """
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def require_keys(
    data: Mapping[str, object], keys: Sequence[str], *, label: str
) -> None:
    """Check that a payload holds every key it is read for.

    Args:
        data: The payload being read.
        keys: The keys that must be present.
        label: What the payload is, for the message.

    Raises:
        ConfigError: If any key is absent. A truncated payload is
            refused rather than defaulted: a default would silently
            stand in for a decision the payload no longer records.

    """
    missing = [key for key in keys if key not in data]
    if missing:
        logger.warning("Rejecting a {} missing key(s): {}", label, missing)
        raise ConfigError(f"The {label} is missing key(s): {missing}")


def mapping_from_payload(value: object, *, label: str) -> Mapping[str, object]:
    """Read a nested object out of a payload.

    Args:
        value: The candidate value, of any type.
        label: What the value is, for the message.

    Returns:
        ``value`` as a mapping.

    Raises:
        ConfigError: If ``value`` is not a mapping.

    """
    if not isinstance(value, Mapping):
        logger.warning("Rejecting a non-mapping {}: {}", label, type(value).__name__)
        raise ConfigError(f"The {label} must be an object, got {type(value).__name__}")
    return value


def duration_from_payload(value: object, *, label: str) -> Duration:
    """Read one duration out of a payload.

    Args:
        value: The candidate value, of any type.
        label: What the duration is, for the message.

    Returns:
        The parsed Duration.

    Raises:
        ConfigError: If ``value`` is not a string, or is a string
            outside the compact duration grammar.

    """
    if not isinstance(value, str):
        logger.warning(
            "Rejecting a non-string {} duration: {}", label, type(value).__name__
        )
        raise ConfigError(
            f"The {label} must be a compact duration string, got {type(value).__name__}"
        )
    return Duration.parse(value)


def optional_duration_from_payload(value: object, *, label: str) -> Duration | None:
    """Read a duration that a payload is allowed to leave unstated.

    Args:
        value: The candidate value, of any type. ``None`` means the
            duration was not stated.
        label: What the duration is, for the message.

    Returns:
        The parsed Duration, or None.

    Raises:
        ConfigError: If ``value`` is neither None nor a valid duration
            string.

    """
    if value is None:
        return None
    return duration_from_payload(value, label=label)


def durations_from_payload(value: object, *, label: str) -> tuple[Duration, ...]:
    """Read a list of durations out of a payload.

    Args:
        value: The candidate value, of any type.
        label: What the list is, for the message.

    Returns:
        The parsed Durations, in stored order.

    Raises:
        ConfigError: If ``value`` is not a list, or if any element is
            not a valid duration string. A bare string is refused
            explicitly: it is iterable, and iterating it would read one
            character per window.

    """
    if not isinstance(value, list | tuple):
        logger.warning(
            "Rejecting a {} that is not a list: {}", label, type(value).__name__
        )
        raise ConfigError(
            f"The {label} must be a list of compact duration strings, got "
            f"{type(value).__name__}"
        )
    return tuple(duration_from_payload(item, label=label) for item in value)


def optional_text_from_payload(value: object, *, label: str) -> str | None:
    """Read a name that a payload is allowed to leave unstated.

    Args:
        value: The candidate value, of any type.
        label: What the text is, for the message.

    Returns:
        ``value`` as a string, or None.

    Raises:
        ConfigError: If ``value`` is neither None nor a string.

    """
    if value is None or isinstance(value, str):
        return value
    logger.warning("Rejecting a non-string {}: {}", label, type(value).__name__)
    raise ConfigError(f"The {label} must be a string, got {type(value).__name__}")


def require_recorded_id(recorded: object, derived: str, *, label: str) -> None:
    """Check a payload's recorded content hash against the one it implies.

    This is what makes a hand-edited payload fail loudly. The realistic
    corruption is an edit to a stored recipe -- a window changed, a
    bound nudged -- without re-deriving the id beside it, and reading
    that back as a valid schedule would attach an id to something it
    does not describe. The guarantee covers the fields an object is
    rebuilt FROM: derived fields such as a recorded limiting ratio are
    recomputed on read rather than verified, so an edit to one of those
    is silently replaced by the recomputed value, not detected.

    Args:
        recorded: The id the payload carries, of any type.
        derived: The id the reconstructed object computes for itself.
        label: The name of the id field, for the message.

    Raises:
        ConfigError: If the recorded id does not match. A value of the
            wrong type fails the same comparison -- nothing but the
            matching string equals the derived one -- so there is no
            separate type check to get out of step with it.

    """
    if recorded != derived:
        logger.warning(
            "Rejecting a payload whose {} does not match its contents.", label
        )
        raise ConfigError(
            f"The recorded {label} does not match the payload it names: stored "
            f"{bounded_echo(str(recorded))}, derived {derived!r}. The payload was "
            "edited without re-deriving its id."
        )
