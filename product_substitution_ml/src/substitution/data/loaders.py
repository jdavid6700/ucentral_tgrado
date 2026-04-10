from __future__ import annotations
from pathlib import Path
from src.substitution.utils.io import read_table


def load_raw_input(raw_dir: str, filename: str):
    path = Path(raw_dir) / filename
    return read_table(str(path))
