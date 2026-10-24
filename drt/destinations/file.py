"""File destination — write records to CSV, JSON, or JSONL files.

No extra dependencies required (uses stdlib csv/json + built-in I/O).

`run_sync()` calls `load()` once per `sync.batch_size` rows on the same
instance, so the first batch of a run truncates the target file and the later
batches append to it.

Example sync YAML:

    destination:
      type: file
      path: output/users.csv
      format: csv

    destination:
      type: file
      path: output/users.json
      format: json

    destination:
      type: file
      path: output/users.jsonl
      format: jsonl
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any

from drt.config.models import DestinationConfig, FileDestinationConfig, SyncOptions
from drt.destinations.base import SyncResult


class FileDestination:
    """Write records to a CSV, JSON, or JSONL file."""

    def __init__(self) -> None:
        # Write state for one run of a sync. The engine calls load() once per
        # batch on this same instance, so the first batch truncates the file and
        # the later batches append to it.
        self._csv_columns: dict[str, tuple[str, ...]] = {}
        self._json_records: dict[str, list[dict[str, Any]]] = {}
        self._jsonl_started_paths: set[str] = set()

    def load(
        self,
        records: list[dict[str, Any]],
        config: DestinationConfig,
        sync_options: SyncOptions,
    ) -> SyncResult:
        assert isinstance(config, FileDestinationConfig)
        if not records:
            return SyncResult()

        result = SyncResult()

        try:
            os.makedirs(os.path.dirname(config.path) or ".", exist_ok=True)

            if config.format == "csv":
                self._write_csv(config.path, records)
            elif config.format == "json":
                self._write_json(config.path, records)
            elif config.format == "jsonl":
                self._write_jsonl(config.path, records)

            result.success = len(records)
        except Exception as e:
            result.failed = len(records)
            result.errors.append(str(e))

        return result

    def reset_write_state(
        self,
        config: DestinationConfig,
        sync_options: SyncOptions,
    ) -> None:
        """Drop this run's write state.

        `run_sync()` calls this after the batch loop, on every exit path. A
        library caller that reuses one instance for several `run_sync()` calls
        must not let the earlier run's state append to the next run's file.
        The call is pure state reset, so it is safe to repeat.
        """
        del config, sync_options
        self._csv_columns.clear()
        self._json_records.clear()
        self._jsonl_started_paths.clear()

    def _write_csv(self, path: str, records: list[dict[str, Any]]) -> None:
        columns = self._csv_columns.get(path)
        first_batch = columns is None
        batch_columns = tuple(dict.fromkeys(key for record in records for key in record))
        if columns is None:
            # No header is written yet, so every field of this batch can
            # contribute a column. dict.fromkeys keeps the first-seen order.
            columns = batch_columns

        unexpected = sorted(set(batch_columns) - set(columns))
        if unexpected:
            # The rows of an earlier batch are already positioned under
            # `columns`. A wider row would misalign or drop data, so fail before
            # the file is opened.
            raise ValueError(
                f"CSV columns cannot change after the header was written for "
                f"'{path}': expected {list(columns)!r}; unexpected {unexpected!r}"
            )

        mode = "w" if first_batch else "a"
        with open(path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=columns,
                restval="",
                extrasaction="raise",
            )
            if first_batch:
                writer.writeheader()
            writer.writerows(records)
        if first_batch:
            self._csv_columns[path] = columns

    def _write_json(self, path: str, records: list[dict[str, Any]]) -> None:
        # A single top-level JSON array cannot be extended with a plain append.
        # Buffer this format's run in memory and rewrite the valid array on each
        # batch. The memory tradeoff is specific to array JSON: CSV and JSONL
        # stay streaming and keep only small bookkeeping state.
        accumulated = [*self._json_records.get(path, []), *records]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(accumulated, f, indent=2, default=str)
        self._json_records[path] = accumulated

    def _write_jsonl(self, path: str, records: list[dict[str, Any]]) -> None:
        mode = "a" if path in self._jsonl_started_paths else "w"
        with open(path, mode, encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")
        self._jsonl_started_paths.add(path)
