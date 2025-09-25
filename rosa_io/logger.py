"""Lightweight structured logger for ROSA pipeline."""

from __future__ import annotations

import json
import logging
import logging.handlers
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "get_logger",
    "setup_file_logger",
    "StructuredAdapter",
]


class StructuredAdapter(logging.LoggerAdapter):
    """Logger adapter that merges structured context with every message."""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        extra = kwargs.get("extra", {})
        context = {**self.extra, **extra}
        if context:
            msg = json.dumps({"message": msg, **context})
            kwargs["extra"] = {}
        return msg, kwargs


def get_logger(name: str = "rosa") -> StructuredAdapter:
    base_logger = logging.getLogger(name)
    return StructuredAdapter(base_logger, {})


@dataclass
class FileLoggerConfig:
    path: Path
    level: int = logging.INFO
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 3
    fmt: str = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"

    def create_handler(self) -> logging.Handler:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            self.path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(self.fmt, self.datefmt))
        handler.setLevel(self.level)
        return handler


def setup_file_logger(config: FileLoggerConfig, name: str = "rosa") -> StructuredAdapter:
    logger = logging.getLogger(name)
    logger.setLevel(config.level)
    handler = config.create_handler()
    logger.addHandler(handler)
    return StructuredAdapter(logger, {})
