"""Parquet file destination — write records to a local Parquet file.

Uses pandas + pyarrow for Parquet serialisation. Supports partitioning
and compression (snappy, gzip, zstd, none).

Requires: pip install drt-core[parquet]

Example sync YAML:

    destination:
      type: parquet
      path: output/scores.parquet
      compression: snappy
      partition_by: [region]
"""

from __future__ import annotations

import os
from typing import Any

from drt.config.models import DestinationConfig, ParquetDestinationConfig, SyncOptions
from drt.destinations.base import SyncResult


class ParquetDestination:
    """Write records to a Parquet file."""

    def __init__(self) -> None:
        # Write state for one run of a sync. The engine calls load() once per
        # batch on this same instance, so each batch rewrites the file with
        # every row of the run.
        self._accumulated: dict[str, list[dict[str, Any]]] = {}

    def load(
        self,
        records: list[dict[str, Any]],
        config: DestinationConfig,
        sync_options: SyncOptions,
    ) -> SyncResult:
        assert isinstance(config, ParquetDestinationConfig)
        if not records:
            return SyncResult()

        try:
            import pandas as pd  # type: ignore[import-untyped]
            import pyarrow  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Parquet destination requires: pip install drt-core[parquet]"
            ) from e

        result = SyncResult()

        try:
            # A Parquet file holds one table, so every batch of a run must be
            # written with the rows that earlier batches already wrote. Buffer
            # this path's run in memory and rewrite the table on each batch.
            accumulated = [*self._accumulated.get(config.path, []), *records]
            df = pd.DataFrame(accumulated)
            os.makedirs(os.path.dirname(config.path) or ".", exist_ok=True)

            compression = config.compression if config.compression != "none" else None

            df.to_parquet(
                config.path,
                engine="pyarrow",
                compression=compression,
                index=False,
                partition_cols=config.partition_by,
            )
            self._accumulated[config.path] = accumulated
            result.success = len(records)
        except Exception as e:
            result.failed = len(records)
            result.errors.append(str(e))
            return result

        return result

    def reset_write_state(
        self,
        config: DestinationConfig,
        sync_options: SyncOptions,
    ) -> None:
        """Drop this run's write state.

        `run_sync()` calls this after the batch loop, on every exit path. A
        library caller that reuses one instance for several `run_sync()` calls
        must not let the earlier run's rows reach the next run's file. The call
        is pure state reset, so it is safe to repeat.
        """
        del config, sync_options
        self._accumulated.clear()
