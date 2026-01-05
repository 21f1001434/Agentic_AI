from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import time
import hashlib

import pandas as pd

from config import Settings
from db import run_sql_query
from cache.snapshot_cache import SnapshotCache  # your existing cache module
from cache.duckdb_store import DuckDBStore


@dataclass
class Executor:
    settings: Settings

    def __post_init__(self) -> None:
        self.cache = SnapshotCache(Path(self.settings.CACHE_DIR))
        self.duckdb = DuckDBStore(Path(self.settings.DUCKDB_PATH))

    def _cache_key(self, sql: str, params: Dict[str, Any]) -> str:
        payload = (sql + "|" + repr(sorted((params or {}).items()))).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def run(self, *, sql: str, params: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes SQL safely (SELECT-only assumed already validated).
        Uses Parquet snapshot caching.
        Registers snapshots into DuckDB catalog for offline querying.
        """
        start = time.time()
        cache_key = self._cache_key(sql, params or {})

        # Try cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            df = cached
            parquet_path = self.cache.path_for_key(cache_key)
            if parquet_path:
                self.duckdb.register_parquet(cache_key, parquet_path)
            meta = {
                "cache_key": cache_key,
                "cache_hit": True,
                "rows": int(len(df)),
                "seconds": round(time.time() - start, 4),
                "mode": "cache",
            }
            return df, meta

        # Offline-only mode: do not hit DB
        if bool(getattr(self.settings, "OFFLINE_ONLY", False)):
            raise RuntimeError(
                "OFFLINE_ONLY is enabled and no cache snapshot exists for this query. "
                "Run once online (or create snapshots) to populate cache."
            )

        # Execute against DB
        df = run_sql_query(
            sql=sql,
            params=params or {},
            timeout_seconds=int(self.settings.QUERY_TIMEOUT_SECONDS),
            max_rows=int(self.settings.MAX_RETURNED_ROWS),
        )

        # Cache to parquet
        self.cache.put(cache_key, df)
        parquet_path = self.cache.path_for_key(cache_key)
        if parquet_path:
            self.duckdb.register_parquet(cache_key, parquet_path)

        meta = {
            "cache_key": cache_key,
            "cache_hit": False,
            "rows": int(len(df)),
            "seconds": round(time.time() - start, 4),
            "mode": "db",
        }
        return df, meta