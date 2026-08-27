"""Logger definitions and configuration."""

import os
import sys
from datetime import datetime

import orjson
from loguru._logger import Core as _Core
from loguru._logger import Logger as _Logger

STDOUT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
FILE_LOG_LEVEL = os.getenv("FILE_LOG_LEVEL", "DEBUG").upper()
ENABLE_FILE_LOGGING = os.getenv("ENABLE_FILE_LOGGING", "").upper() == "TRUE"
DISABLE_COLORIZE_LOGS = os.getenv("DISABLE_COLORIZE_LOGS", "").upper() == "TRUE"
ENABLE_SERIALIZE_LOGS = os.getenv("ENABLE_SERIALIZE_LOGS", "").upper() == "TRUE"
LOGURU_DIAGNOSE = os.getenv("LOGURU_DIAGNOSE", "").upper() == "TRUE"

colorize = not DISABLE_COLORIZE_LOGS
serialize = ENABLE_SERIALIZE_LOGS

# Create a mapping of module name to color
color_map = {
    "ohlc_toolkit": "blue",
}


# This will hold our logger instances
loggers: dict[str, _Logger] = {}


# Export this logger
def get_logger(name: str) -> _Logger:
    """Get a logger instance for the given name."""
    main_module_name = _extract_main_module_name(name)

    # Check if a logger for this name already exists
    if main_module_name in loggers:
        return loggers[main_module_name].bind(name=name)

    logger_ = _create_logger_instance()

    _configure_logger(logger_)

    if not serialize:
        _setup_stdout_logging(logger_, main_module_name)
    else:
        _setup_serialized_logging(logger_)

    if ENABLE_FILE_LOGGING:
        _setup_file_logging(logger_, main_module_name, name)

    loggers[main_module_name] = logger_

    return logger_.bind(name=name)


def _extract_main_module_name(name: str) -> str:
    return name.split(".", maxsplit=1)[0]


def _create_logger_instance() -> _Logger:
    return _Logger(
        core=_Core(),
        exception=None,
        depth=0,
        record=False,
        lazy=False,
        colors=False,
        raw=False,
        capture=True,
        patchers=[],
        extra={},
    )


def _configure_logger(logger_: _Logger):
    logger_.configure(extra={"body": ""})


# Define custom formatter for this module
def _formatter_builder(color: str):
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        f"<{color}>{{name}}</{color}>:<{color}>{{function}}</{color}>:<{color}>{{line}}</{color}> | "
        "<level>{message}</level> | "
        "{extra[body]}"
    )


# Define custom formatter for serialized logs
def _serialize_record(record):
    record_time: datetime = record["time"]

    # format the time field to ISO8601 format
    iso_date = record_time.isoformat()

    # Handle exceptions as default
    exception = record["exception"]

    if exception is not None:
        exception = {
            "type": None if exception.type is None else exception.type.__name__,
            "value": exception.value,
            "traceback": bool(exception.traceback),
        }

    # Define subset of serialized record - combining message + extra into the text field
    message = record["message"]
    extra = record["extra"].get("body")
    message_with_body = f"{message} | {extra}"

    # we keep fields commented out to compare with loguru's default serialisation
    subset = {
        "message": message_with_body,
        "levelname": record["level"].name,  # log level
        "date": iso_date,
        "name": record["name"],
        "record": {
            # "elapsed": {
            #     "repr": record["elapsed"],
            #     "seconds": record["elapsed"].total_seconds(),
            # },
            "exception": exception,
            # "extra": record["extra"],
            # "file": {"name": record["file"].name, "path": record["file"].path},
            "file": record["file"].path,
            "function": record["function"],
            # "level": {
            #     "icon": record["level"].icon,
            #     "name": record["level"].name,
            #     "no": record["level"].no,
            # },
            "line": record["line"],
            # "message": record["message"],
            # "module": record["module"],
            "process": {"id": record["process"].id, "name": record["process"].name},
            "thread": {"id": record["thread"].id, "name": record["thread"].name},
            "time": {
                "repr": int(1000 * record_time.timestamp()),  # to milliseconds
                "uptime_h:m:s": record["elapsed"],
            },
        },
    }
    record["extra"]["serialized"] = orjson.dumps(subset, default=str).decode("utf-8")
    return "{extra[serialized]}\n"


def _setup_stdout_logging(logger_: _Logger, main_module_name: str):
    color = color_map.get(main_module_name, "blue")
    formatter = _formatter_builder(color)
    logger_.add(
        sys.stdout,
        level=STDOUT_LOG_LEVEL,
        diagnose=True,
        format=formatter,
        colorize=colorize,
    )


def _setup_serialized_logging(logger_: _Logger):
    logger_.add(
        sys.stdout,
        level=STDOUT_LOG_LEVEL,
        diagnose=LOGURU_DIAGNOSE,
        format=_serialize_record,
    )


def _get_log_file_path(main_module_name) -> str:
    # The absolute path of this file's directory
    config_dir = os.path.dirname(os.path.abspath(__file__))

    # Move up three levels to get to the project root directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(config_dir)))

    # Define the logging dir with
    log_dir = os.path.join(base_dir, f"logs/{main_module_name}")
    return os.path.join(log_dir, "{time:YYYY-MM-DD}.log")


def _setup_file_logging(logger_: _Logger, main_module_name: str, name: str):
    try:
        logger_.add(
            _get_log_file_path(main_module_name),
            rotation="00:00",
            retention="7 days",
            enqueue=True,
            level=FILE_LOG_LEVEL,
            diagnose=True,
            format=_formatter_builder("blue"),
            serialize=serialize,
        )
    except PermissionError:
        logger_.warning(
            "Permission error caught when trying to create log file. "
            "Continuing without file logging for `{}` in `{}`",
            name,
            main_module_name,
        )
