"""Logging setup. Plain text by default; JSON when config.logging.json is true."""

from __future__ import annotations

import json
import logging
import sys
import time


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


def setup_logging(
    level: str = "INFO", json_output: bool = False, log_file: str | None = None
) -> None:
    # The dashboard owns the terminal, so route logs to a file when one is given.
    handler: logging.Handler = (
        logging.FileHandler(log_file, encoding="utf-8")
        if log_file
        else logging.StreamHandler(sys.stderr)
    )
    if json_output:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
        )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    logging.Formatter.converter = time.localtime


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
