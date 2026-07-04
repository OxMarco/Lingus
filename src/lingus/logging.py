"""Logging setup. Plain text by default; JSON when config.logging.json is true.

On an interactive terminal the plain-text lines are colored by severity so the
firehose is easy to scan: debug is dim/white, info blue, warning yellow, and
error/critical red. Colors are suppressed when logging to a file, when the
stream is not a TTY (piped/redirected), or in JSON mode, so machine-readable
output stays clean.
"""

from __future__ import annotations

import json
import logging
import sys
import time

# ANSI SGR codes keyed by log level. CRITICAL gets bold-red to stand apart.
_RESET = "\033[0m"
_LEVEL_COLORS = {
    logging.DEBUG: "\033[37m",  # white
    logging.INFO: "\033[34m",  # blue
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


class _ColorFormatter(logging.Formatter):
    """Wrap each formatted line in the ANSI color for its level."""

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        color = _LEVEL_COLORS.get(record.levelno)
        return f"{color}{line}{_RESET}" if color else line


def setup_logging(
    level: str = "INFO", json_output: bool = False, log_file: str | None = None
) -> None:
    # The dashboard owns the terminal, so route logs to a file when one is given.
    handler: logging.Handler = (
        logging.FileHandler(log_file, encoding="utf-8")
        if log_file
        else logging.StreamHandler(sys.stderr)
    )
    fmt = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    if json_output:
        handler.setFormatter(_JsonFormatter())
    elif log_file is None and getattr(handler.stream, "isatty", lambda: False)():
        # Interactive terminal: color by severity for at-a-glance scanning.
        handler.setFormatter(_ColorFormatter(fmt))
    else:
        # File or piped output: keep it plain and grep-friendly.
        handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    logging.Formatter.converter = time.localtime

    # Chatty third-party loggers log every HTTP request at INFO (e.g. httpx
    # narrating each Hugging Face model-metadata fetch). Pin them to WARNING so
    # they don't drown out our own INFO lines; DEBUG re-enables them explicitly.
    third_party_level = logging.DEBUG if level.upper() == "DEBUG" else logging.WARNING
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(third_party_level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
