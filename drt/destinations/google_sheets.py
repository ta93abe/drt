"""Google Sheets destination — write rows to a spreadsheet.

Requires: pip install drt-core[sheets]

Example sync YAML:

    destination:
      type: google_sheets
      spreadsheet_id: "1BxiMVs..."
      sheet: "Sheet1"
      mode: overwrite
"""

from __future__ import annotations

import os
from typing import Any

from drt.config.models import DestinationConfig, GoogleSheetsDestinationConfig, SyncOptions
from drt.destinations.base import SyncResult


def _build_sheets_service(config: GoogleSheetsDestinationConfig) -> Any:
    """Build Google Sheets API v4 service client."""
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    keyfile = config.credentials_path or (
        os.environ.get(config.credentials_env) if config.credentials_env else None
    )

    if keyfile:
        creds = service_account.Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
            keyfile, scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
    else:
        import google.auth

        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )

    return build("sheets", "v4", credentials=creds)


class GoogleSheetsDestination:
    """Write records to a Google Sheets spreadsheet."""

    def __init__(self) -> None:
        # Write state for one run of a sync. The engine calls load() once per
        # batch on this same instance, so the first batch of a run writes the
        # header and the later batches append their rows to it.
        self._headers: list[str] | None = None

    def load(
        self,
        records: list[dict[str, Any]],
        config: DestinationConfig,
        sync_options: SyncOptions,
    ) -> SyncResult:
        assert isinstance(config, GoogleSheetsDestinationConfig)
        if not records:
            return SyncResult()

        result = SyncResult()

        try:
            service = _build_sheets_service(config)
            sheets = service.spreadsheets()
            range_name = config.sheet

            first_batch = self._headers is None
            headers = (
                self._headers
                if self._headers is not None
                else list(dict.fromkeys(key for record in records for key in record))
            )
            unexpected = sorted({key for record in records for key in record} - set(headers))
            if unexpected:
                # The rows of an earlier batch of this run are already positioned
                # under `headers`. A wider row would misalign or drop data, so
                # fail before the sheet is touched.
                raise ValueError(
                    f"Sheet columns cannot change after the header was written for "
                    f"'{range_name}': expected {headers!r}; unexpected {unexpected!r}"
                )
            rows = [[str(row.get(h, "")) for h in headers] for row in records]

            if config.mode == "overwrite" and first_batch:
                # Clear the sheet once per run, then write the header and the
                # first batch.
                sheets.values().clear(
                    spreadsheetId=config.spreadsheet_id,
                    range=range_name,
                    body={},
                ).execute()
                sheets.values().update(
                    spreadsheetId=config.spreadsheet_id,
                    range=range_name,
                    valueInputOption="RAW",
                    body={"values": [headers, *rows]},
                ).execute()
            else:
                # Later batches of a run append their rows under the header that
                # this run wrote.
                sheets.values().append(
                    spreadsheetId=config.spreadsheet_id,
                    range=range_name,
                    valueInputOption="RAW",
                    body={"values": rows},
                ).execute()

            self._headers = headers
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
        must not let the earlier run's header reach the next run's sheet. The
        call is pure state reset, so it is safe to repeat.
        """
        del config, sync_options
        self._headers = None
