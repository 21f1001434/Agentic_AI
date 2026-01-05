from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class SnapshotCache:
    """
    Local snapshot cache for query results.

    Stores each query result as a Parquet file:
      cache/<cache_key>.parquet

    This avoids re-querying the DB for repeated analytics/dashboard runs.

    NOTE: This cache is local only; it does NOT alter the source database.
    """

    cache_dir: Path

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for_key(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.parquet"

    def get(self, cache_key: str) -> Optional[pd.DataFrame]:
        path = self.path_for_key(cache_key)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            # corrupt cache file → ignore (safe fallback)
            return None

    def put(self, cache_key: str, df: pd.DataFrame) -> Path:
        path = self.path_for_key(cache_key)
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write Parquet (fast and compact)
        df.to_parquet(path, index=False)
        return path

    def delete(self, cache_key: str) -> bool:
        path = self.path_for_key(cache_key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear_all(self) -> int:
        """
        Deletes all cached parquet files.
        Returns number of deleted files.
        """
        n = 0
        for p in self.cache_dir.glob("*.parquet"):
            try:
                p.unlink()
                n += 1
            except Exception:
                pass
        return n