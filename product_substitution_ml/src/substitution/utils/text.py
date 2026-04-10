from __future__ import annotations
import re
import unicodedata


def normalize_text(value) -> str:
    if value is None:
        return ""
    value = str(value).strip().upper()
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("utf-8")
    value = re.sub(r"\s+", " ", value)
    return value


def safe_str(value) -> str:
    return "" if value is None else str(value).strip()
