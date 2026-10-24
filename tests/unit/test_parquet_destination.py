"""Unit tests for Parquet file destination.

Uses tmp_path for real file writes — no mocking needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from drt.config.models import ParquetDestinationConfig, SyncOptions
from drt.destinations.parquet import ParquetDestination

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _options(**kwargs: Any) -> SyncOptions:
    return SyncOptions(**kwargs)


def _config(tmp_path: Path, **overrides: Any) -> ParquetDestinationConfig:
    defaults: dict[str, Any] = {
        "type": "parquet",
        "path": str(tmp_path / "output.parquet"),
    }
    defaults.update(overrides)
    return ParquetDestinationConfig(**defaults)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestParquetDestinationConfig:
    def test_valid_config(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        assert config.type == "parquet"
        assert config.compression == "snappy"
        assert config.partition_by is None

    def test_custom_compression(self, tmp_path: Path) -> None:
        config = _config(tmp_path, compression="gzip")
        assert config.compression == "gzip"

    def test_partition_by(self, tmp_path: Path) -> None:
        config = _config(tmp_path, partition_by=["region", "year"])
        assert config.partition_by == ["region", "year"]

    def test_invalid_compression_rejected(self, tmp_path: Path) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="compression"):
            _config(tmp_path, compression="invalid")


# ---------------------------------------------------------------------------
# Load behavior
# ---------------------------------------------------------------------------


class TestParquetDestinationLoad:
    def test_success_write(self, tmp_path: Path) -> None:
        records = [
            {"id": 1, "score": 0.95, "region": "north"},
            {"id": 2, "score": 0.80, "region": "south"},
        ]
        config = _config(tmp_path)
        result = ParquetDestination().load(records, config, _options())

        assert result.success == 2
        assert result.failed == 0
        assert Path(config.path).exists()

    def test_empty_records(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        result = ParquetDestination().load([], config, _options())

        assert result.success == 0
        assert result.failed == 0
        assert not Path(config.path).exists()

    def test_file_content_readable(self, tmp_path: Path) -> None:
        import pandas as pd

        records = [
            {"id": 1, "name": "alice", "value": 100},
            {"id": 2, "name": "bob", "value": 200},
        ]
        config = _config(tmp_path)
        ParquetDestination().load(records, config, _options())

        df = pd.read_parquet(config.path)
        assert len(df) == 2
        assert list(df.columns) == ["id", "name", "value"]
        assert df["value"].sum() == 300

    def test_gzip_compression(self, tmp_path: Path) -> None:
        records = [{"id": 1, "data": "test"}]
        config = _config(tmp_path, compression="gzip")
        result = ParquetDestination().load(records, config, _options())

        assert result.success == 1
        assert Path(config.path).exists()

    def test_no_compression(self, tmp_path: Path) -> None:
        records = [{"id": 1, "data": "test"}]
        config = _config(tmp_path, compression="none")
        result = ParquetDestination().load(records, config, _options())

        assert result.success == 1
        assert Path(config.path).exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        deep_path = str(tmp_path / "a" / "b" / "c" / "output.parquet")
        config = _config(tmp_path, path=deep_path)
        records = [{"id": 1}]
        result = ParquetDestination().load(records, config, _options())

        assert result.success == 1
        assert Path(deep_path).exists()

    def test_partition_by_creates_directories(self, tmp_path: Path) -> None:
        out_dir = str(tmp_path / "partitioned")
        config = _config(tmp_path, path=out_dir, partition_by=["region"])
        records = [
            {"id": 1, "region": "north", "value": 10},
            {"id": 2, "region": "south", "value": 20},
        ]
        result = ParquetDestination().load(records, config, _options())

        assert result.success == 2
        # Partitioned parquet creates subdirectories
        assert Path(out_dir).exists()

    def test_error_returns_failure(self, tmp_path: Path) -> None:
        # Invalid path that can't be written
        config = _config(tmp_path, path="")
        records = [{"id": 1}]
        result = ParquetDestination().load(records, config, _options())

        assert result.failed == len(records)
        assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# Multi-batch runs — run_sync() calls load() once per batch on one instance
# ---------------------------------------------------------------------------


class TestParquetDestinationMultiBatch:
    def test_keeps_every_batch_in_one_file(self, tmp_path: Path) -> None:
        import pandas as pd

        config = _config(tmp_path)
        dest = ParquetDestination()

        results = [
            dest.load([{"id": 1}, {"id": 2}], config, _options()),
            dest.load([{"id": 3}, {"id": 4}], config, _options()),
            dest.load([{"id": 5}], config, _options()),
        ]

        assert list(pd.read_parquet(config.path)["id"]) == [1, 2, 3, 4, 5]
        assert [result.success for result in results] == [2, 2, 1]

    def test_reset_write_state_starts_a_new_run(self, tmp_path: Path) -> None:
        import pandas as pd

        config = _config(tmp_path)
        dest = ParquetDestination()
        dest.load([{"id": 1}, {"id": 2}], config, _options())

        dest.reset_write_state(config, _options())
        dest.load([{"id": 3}], config, _options())

        assert list(pd.read_parquet(config.path)["id"]) == [3]
