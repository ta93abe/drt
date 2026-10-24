"""Unit tests for CSV/JSON/JSONL file destination.

Uses tmp_path for real file writes — no mocking needed, no extra dependencies.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from drt.config.models import FileDestinationConfig, SyncOptions
from drt.destinations.file import FileDestination

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _options(**kwargs: Any) -> SyncOptions:
    return SyncOptions(**kwargs)


def _config(tmp_path: Path, **overrides: Any) -> FileDestinationConfig:
    defaults: dict[str, Any] = {
        "type": "file",
        "path": str(tmp_path / "output.csv"),
        "format": "csv",
    }
    defaults.update(overrides)
    return FileDestinationConfig(**defaults)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestFileDestinationConfig:
    def test_valid_csv_config(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        assert config.format == "csv"

    def test_json_format(self, tmp_path: Path) -> None:
        config = _config(tmp_path, format="json")
        assert config.format == "json"

    def test_jsonl_format(self, tmp_path: Path) -> None:
        config = _config(tmp_path, format="jsonl")
        assert config.format == "jsonl"

    def test_invalid_format_rejected(self, tmp_path: Path) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="format"):
            _config(tmp_path, format="xml")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


class TestCsvDestination:
    def test_csv_write(self, tmp_path: Path) -> None:
        records = [
            {"id": 1, "name": "alice", "score": 95},
            {"id": 2, "name": "bob", "score": 80},
        ]
        config = _config(tmp_path)
        result = FileDestination().load(records, config, _options())

        assert result.success == 2
        assert result.failed == 0
        assert Path(config.path).exists()

    def test_csv_content_readable(self, tmp_path: Path) -> None:
        records = [
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
        ]
        config = _config(tmp_path)
        FileDestination().load(records, config, _options())

        with open(config.path, newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["name"] == "alice"
        assert reader[1]["name"] == "bob"

    def test_csv_creates_parent_dirs(self, tmp_path: Path) -> None:
        deep_path = str(tmp_path / "a" / "b" / "output.csv")
        config = _config(tmp_path, path=deep_path)
        result = FileDestination().load([{"id": 1}], config, _options())

        assert result.success == 1
        assert Path(deep_path).exists()


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


class TestJsonDestination:
    def test_json_write(self, tmp_path: Path) -> None:
        records = [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
        config = _config(tmp_path, format="json", path=str(tmp_path / "out.json"))
        result = FileDestination().load(records, config, _options())

        assert result.success == 2
        with open(config.path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["name"] == "alice"

    def test_json_handles_non_serializable(self, tmp_path: Path) -> None:
        from datetime import datetime

        records = [{"id": 1, "ts": datetime(2026, 1, 1)}]
        config = _config(tmp_path, format="json", path=str(tmp_path / "out.json"))
        result = FileDestination().load(records, config, _options())

        assert result.success == 1  # default=str handles it


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


class TestJsonlDestination:
    def test_jsonl_write(self, tmp_path: Path) -> None:
        records = [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
        config = _config(tmp_path, format="jsonl", path=str(tmp_path / "out.jsonl"))
        result = FileDestination().load(records, config, _options())

        assert result.success == 2
        with open(config.path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) == 2
        assert lines[1]["name"] == "bob"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestFileDestinationEdgeCases:
    def test_empty_records(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        result = FileDestination().load([], config, _options())

        assert result.success == 0
        assert result.failed == 0
        assert not Path(config.path).exists()

    def test_error_returns_failure(self, tmp_path: Path) -> None:
        config = _config(tmp_path, path="")
        result = FileDestination().load([{"id": 1}], config, _options())

        assert result.failed == 1
        assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# Multi-batch runs — run_sync() calls load() once per batch on one instance
# ---------------------------------------------------------------------------


class TestFileDestinationMultiBatch:
    def test_csv_keeps_every_batch_in_one_file(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        dest = FileDestination()

        results = [
            dest.load([{"id": 1}, {"id": 2}], config, _options()),
            dest.load([{"id": 3}, {"id": 4}], config, _options()),
            dest.load([{"id": 5}], config, _options()),
        ]

        with open(config.path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert [row["id"] for row in rows] == ["1", "2", "3", "4", "5"]
        assert [result.success for result in results] == [2, 2, 1]
        # The header is written once, not once per batch.
        with open(config.path, encoding="utf-8") as f:
            assert f.read().splitlines().count("id") == 1

    def test_json_keeps_every_batch_in_one_array(self, tmp_path: Path) -> None:
        config = _config(tmp_path, path=str(tmp_path / "output.json"), format="json")
        dest = FileDestination()

        dest.load([{"id": 1}, {"id": 2}], config, _options())
        dest.load([{"id": 3}], config, _options())

        with open(config.path, encoding="utf-8") as f:
            assert json.load(f) == [{"id": 1}, {"id": 2}, {"id": 3}]

    def test_jsonl_keeps_every_batch(self, tmp_path: Path) -> None:
        config = _config(tmp_path, path=str(tmp_path / "output.jsonl"), format="jsonl")
        dest = FileDestination()

        dest.load([{"id": 1}, {"id": 2}], config, _options())
        dest.load([{"id": 3}], config, _options())

        with open(config.path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        assert [json.loads(line)["id"] for line in lines] == [1, 2, 3]

    def test_reset_write_state_starts_a_new_run(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        dest = FileDestination()
        dest.load([{"id": 1}, {"id": 2}], config, _options())

        dest.reset_write_state(config, _options())
        dest.load([{"id": 3}], config, _options())

        with open(config.path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert [row["id"] for row in rows] == ["3"]

    def test_csv_rejects_new_columns_after_the_header(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        dest = FileDestination()
        dest.load([{"id": 1}], config, _options())

        result = dest.load([{"id": 2, "extra": "x"}], config, _options())

        assert result.failed == 1
        assert "cannot change" in result.errors[0]
        with open(config.path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert [row["id"] for row in rows] == ["1"]
