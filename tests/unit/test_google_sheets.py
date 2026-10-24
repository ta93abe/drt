"""Unit tests for Google Sheets destination.

Mocks the Google Sheets API client since there is no local server equivalent.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from drt.config.models import GoogleSheetsDestinationConfig, SyncOptions


def _options() -> SyncOptions:
    return SyncOptions()


class TestGoogleSheetsDestination:
    def test_overwrite_clears_and_writes(self) -> None:
        from drt.destinations.google_sheets import GoogleSheetsDestination

        config = GoogleSheetsDestinationConfig(
            type="google_sheets",
            spreadsheet_id="test-id",
            sheet="Sheet1",
            mode="overwrite",
        )
        records = [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
        ]

        mock_service = MagicMock()
        mock_sheets = mock_service.spreadsheets.return_value
        mock_values = mock_sheets.values.return_value
        mock_values.clear.return_value.execute.return_value = {}
        mock_values.update.return_value.execute.return_value = {"updatedRows": 3}

        with patch(
            "drt.destinations.google_sheets._build_sheets_service",
            return_value=mock_service,
        ):
            dest = GoogleSheetsDestination()
            result = dest.load(records, config, _options())

        assert result.success == 2
        assert result.failed == 0
        mock_values.clear.assert_called_once()
        mock_values.update.assert_called_once()

    def test_append_does_not_clear(self) -> None:
        from drt.destinations.google_sheets import GoogleSheetsDestination

        config = GoogleSheetsDestinationConfig(
            type="google_sheets",
            spreadsheet_id="test-id",
            sheet="Sheet1",
            mode="append",
        )
        records = [{"id": 1, "name": "Alice"}]

        mock_service = MagicMock()
        mock_sheets = mock_service.spreadsheets.return_value
        mock_values = mock_sheets.values.return_value
        mock_values.append.return_value.execute.return_value = {
            "updates": {"updatedRows": 1}
        }

        with patch(
            "drt.destinations.google_sheets._build_sheets_service",
            return_value=mock_service,
        ):
            dest = GoogleSheetsDestination()
            result = dest.load(records, config, _options())

        assert result.success == 1
        mock_values.clear.assert_not_called()
        mock_values.append.assert_called_once()

    def test_empty_records(self) -> None:
        from drt.destinations.google_sheets import GoogleSheetsDestination

        config = GoogleSheetsDestinationConfig(
            type="google_sheets",
            spreadsheet_id="test-id",
        )
        dest = GoogleSheetsDestination()
        result = dest.load([], config, _options())
        assert result.success == 0
        assert result.failed == 0

    def test_api_error_reports_failure(self) -> None:
        from drt.destinations.google_sheets import GoogleSheetsDestination

        config = GoogleSheetsDestinationConfig(
            type="google_sheets",
            spreadsheet_id="test-id",
            mode="overwrite",
        )
        records = [{"id": 1, "name": "Alice"}]

        mock_service = MagicMock()
        mock_sheets = mock_service.spreadsheets.return_value
        mock_values = mock_sheets.values.return_value
        mock_values.clear.return_value.execute.side_effect = Exception("API error")

        with patch(
            "drt.destinations.google_sheets._build_sheets_service",
            return_value=mock_service,
        ):
            dest = GoogleSheetsDestination()
            result = dest.load(records, config, _options())

        assert result.failed == 1
        assert result.success == 0


class TestGoogleSheetsDestinationMultiBatch:
    """run_sync() calls load() once per batch on one instance."""

    _records = [
        {"id": 0, "name": "n0"},
        {"id": 1, "name": "n1"},
        {"id": 2, "name": "n2"},
        {"id": 3, "name": "n3"},
        {"id": 4, "name": "n4"},
    ]

    def _config(self, mode: str) -> GoogleSheetsDestinationConfig:
        return GoogleSheetsDestinationConfig(
            type="google_sheets",
            spreadsheet_id="test-id",
            sheet="Sheet1",
            mode=mode,
        )

    def _load_batches(
        self, dest: Any, config: GoogleSheetsDestinationConfig, service: MagicMock
    ) -> list[Any]:
        with patch(
            "drt.destinations.google_sheets._build_sheets_service",
            return_value=service,
        ):
            return [
                dest.load(self._records[0:2], config, _options()),
                dest.load(self._records[2:4], config, _options()),
                dest.load(self._records[4:], config, _options()),
            ]

    def test_overwrite_clears_once_then_appends_later_batches(self) -> None:
        from drt.destinations.google_sheets import GoogleSheetsDestination

        service = MagicMock()
        values = service.spreadsheets.return_value.values.return_value
        dest = GoogleSheetsDestination()

        results = self._load_batches(dest, self._config("overwrite"), service)

        assert [result.success for result in results] == [2, 2, 1]
        values.clear.assert_called_once()
        assert values.update.call_args.kwargs["body"]["values"] == [
            ["id", "name"],
            ["0", "n0"],
            ["1", "n1"],
        ]
        appended = [call.kwargs["body"]["values"] for call in values.append.call_args_list]
        assert appended == [[["2", "n2"], ["3", "n3"]], [["4", "n4"]]]

    def test_append_mode_appends_every_batch(self) -> None:
        from drt.destinations.google_sheets import GoogleSheetsDestination

        service = MagicMock()
        values = service.spreadsheets.return_value.values.return_value
        dest = GoogleSheetsDestination()

        results = self._load_batches(dest, self._config("append"), service)

        assert [result.success for result in results] == [2, 2, 1]
        values.clear.assert_not_called()
        assert [call.kwargs["body"]["values"] for call in values.append.call_args_list] == [
            [["0", "n0"], ["1", "n1"]],
            [["2", "n2"], ["3", "n3"]],
            [["4", "n4"]],
        ]

    def test_reset_write_state_clears_the_sheet_again(self) -> None:
        from drt.destinations.google_sheets import GoogleSheetsDestination

        service = MagicMock()
        values = service.spreadsheets.return_value.values.return_value
        dest = GoogleSheetsDestination()
        config = self._config("overwrite")
        self._load_batches(dest, config, service)

        dest.reset_write_state(config, _options())
        with patch(
            "drt.destinations.google_sheets._build_sheets_service",
            return_value=service,
        ):
            dest.load(self._records[0:2], config, _options())

        assert values.clear.call_count == 2

    def test_later_batch_cannot_add_columns(self) -> None:
        from drt.destinations.google_sheets import GoogleSheetsDestination

        service = MagicMock()
        dest = GoogleSheetsDestination()
        config = self._config("overwrite")
        with patch(
            "drt.destinations.google_sheets._build_sheets_service",
            return_value=service,
        ):
            dest.load([{"id": 0, "name": "n0"}], config, _options())
            result = dest.load([{"id": 1, "name": "n1", "extra": "x"}], config, _options())

        assert result.failed == 1
        assert "cannot change" in result.errors[0]
