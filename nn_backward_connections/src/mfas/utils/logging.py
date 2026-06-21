"""Minimal logging helpers (idempotent logger + structured key=value lines)."""
from __future__ import annotations

import logging

__all__ = ["get_logger", "log_kv"]

_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def get_logger(name: str = "mfas", level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a single StreamHandler (no duplicate handlers)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def log_kv(logger: logging.Logger, msg: str = "", **kv) -> None:
    """Log a one-line ``key=value`` summary."""
    parts = " ".join(f"{k}={v}" for k, v in kv.items())
    logger.info(f"{msg} {parts}".strip())
