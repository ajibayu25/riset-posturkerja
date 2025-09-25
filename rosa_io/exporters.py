"""Simple data exporters for ROSA session logging."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

__all__ = [
    "export_csv",
    "export_json",
    "ensure_parent",
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def export_csv(path: str, row: Mapping[str, object]) -> None:
    file_path = Path(path)
    ensure_parent(file_path)
    row_dict = dict(row)
    with file_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row_dict)


def export_json(path: str, row: Mapping[str, object]) -> None:
    file_path = Path(path)
    ensure_parent(file_path)
    payload = {"ts": datetime.now().isoformat(), **dict(row)}
    with file_path.open("a", encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")
