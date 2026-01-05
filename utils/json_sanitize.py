from __future__ import annotations

from typing import Any
import datetime as dt

import pandas as pd


def json_sanitize(obj: Any) -> Any:
    """
    Convert common non-JSON-serializable objects to safe JSON types.
    - pd.Timestamp / datetime -> ISO string
    - numpy scalars -> python scalars
    - Path -> str
    - set -> list
    """
    # pandas Timestamp
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    # python datetime/date
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()

    # pandas NA / NaT
    if obj is pd.NaT:
        return None

    # numpy scalar fallback (works without importing numpy explicitly)
    tname = type(obj).__name__.lower()
    if "int" in tname or "float" in tname or "bool" in tname:
        try:
            return obj.item()
        except Exception:
            pass

    # bytes
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8", errors="ignore")
        except Exception:
            return str(obj)

    # set
    if isinstance(obj, set):
        return list(obj)

    # fallback
    return str(obj)