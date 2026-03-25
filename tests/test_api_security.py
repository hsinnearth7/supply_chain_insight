"""Security tests: path traversal, auth, CORS, upload limits."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api.routes import (
    WATCHDOG_STAGING_DIR,
    _prune_watchdog_staging,
    _safe_filename,
    _wait_for_stable_csv,
)
from app.config import RAW_DIR
from app.db.models import PipelineRun, SessionLocal


class TestAuthentication:
    """Verify API key authentication is enforced."""

    def test_auth_required_no_key(self, client):
        """Verify 401 without API key."""
        resp = client.get("/api/runs")
        assert resp.status_code == 401

    def test_auth_required_wrong_key(self, client):
        """Verify 403 with invalid API key."""
        resp = client.get("/api/runs", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 403

    def test_auth_success(self, client, auth_headers):
        """Verify 200 with correct API key."""
        resp = client.get("/api/runs", headers=auth_headers)
        assert resp.status_code == 200

    def test_health_no_auth(self, client):
        """Health endpoint should not require auth."""
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_static_charts_not_public(self, client):
        """Chart files should no longer be exposed through a public static mount."""
        resp = client.get("/charts/example/chart.png")
        assert resp.status_code == 404


class TestPathTraversal:
    """Verify path traversal attacks are blocked."""

    def test_chart_traversal_batch_id(self, client, auth_headers):
        """Verify ../../../etc/passwd in batch_id is rejected."""
        resp = client.get("/api/runs/../../../etc/passwd/charts", headers=auth_headers)
        # Should either 404 or 403, not serve the file
        assert resp.status_code in (403, 404, 422)

    def test_chart_traversal_chart_name(self, client, auth_headers):
        """Verify ../../../etc/passwd in chart_name is rejected."""
        resp = client.get(
            "/api/runs/test_batch/charts/../../etc/passwd",
            headers=auth_headers,
        )
        assert resp.status_code in (403, 404)

    def test_chart_name_with_dotdot(self, client, auth_headers):
        """Verify .. in chart name is rejected."""
        resp = client.get(
            "/api/runs/test_batch/charts/..%2F..%2Fetc%2Fpasswd",
            headers=auth_headers,
        )
        assert resp.status_code in (403, 404)


class TestUploadSecurity:
    """Verify upload validations."""

    def test_upload_non_csv_rejected(self, client, auth_headers):
        """Verify non-CSV files are rejected."""
        resp = client.post(
            "/api/ingest",
            headers=auth_headers,
            files={"file": ("test.txt", b"not a csv", "text/plain")},
        )
        assert resp.status_code == 400

    def test_upload_size_limit(self, client, auth_headers):
        """Verify 413 on oversized upload."""
        # Create a file larger than MAX_UPLOAD_SIZE (10MB)
        large_content = b"a,b,c\n" + b"1,2,3\n" * (2 * 1024 * 1024)  # ~12MB
        resp = client.post(
            "/api/ingest",
            headers=auth_headers,
            files={"file": ("big.csv", large_content, "text/csv")},
        )
        assert resp.status_code == 413

    def test_upload_invalid_csv_rejected(self, client, auth_headers):
        """Verify invalid CSV content is rejected."""
        resp = client.post(
            "/api/ingest",
            headers=auth_headers,
            files={"file": ("test.csv", b"\x00\x01\x02binary", "text/csv")},
        )
        assert resp.status_code == 400

    def test_upload_valid_csv_accepted(self, client, auth_headers, sample_csv_content):
        """Verify valid CSV upload is accepted."""
        resp = client.post(
            "/api/ingest",
            headers=auth_headers,
            files={"file": ("test_data.csv", sample_csv_content.encode(), "text/csv")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "batch_id" in data
        assert data["status"] == "queued"

    def test_upload_uppercase_csv_extension_accepted(self, client, auth_headers, sample_csv_content):
        """Verify valid CSV uploads are accepted regardless of extension casing."""
        resp = client.post(
            "/api/ingest",
            headers=auth_headers,
            files={"file": ("DATA.CSV", sample_csv_content.encode(), "text/csv")},
        )
        assert resp.status_code == 200

    def test_ingest_existing_invalid_csv_rejected(self, client, auth_headers):
        """Verify existing-file ingest enforces the same CSV validation rules."""
        invalid_path = RAW_DIR / "SecurityDirty.csv"
        invalid_path.write_bytes(b"\x00\x01\x02binary")
        try:
            resp = client.post("/api/ingest/existing", headers=auth_headers)
        finally:
            if invalid_path.exists():
                invalid_path.unlink()
        assert resp.status_code == 400


class TestWatchdogValidation:
    """Verify watchdog-triggered ingests use the same CSV validation rules."""

    def test_wait_for_stable_csv_rejects_invalid_content(self):
        csv_path = Path("watchdog_invalid_test.csv")
        csv_path.write_bytes(b"\x00\x01\x02binary")
        try:
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(_wait_for_stable_csv(csv_path, checks=2, interval_seconds=0.01))
        finally:
            if csv_path.exists():
                csv_path.unlink()
        assert exc_info.value.status_code == 400

    def test_wait_for_stable_csv_rejects_non_csv_extension(self):
        txt_path = Path("watchdog_invalid_test.txt")
        txt_path.write_text("a,b\n1,2\n", encoding="utf-8")
        try:
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(_wait_for_stable_csv(txt_path, checks=2, interval_seconds=0.01))
        finally:
            if txt_path.exists():
                txt_path.unlink()
        assert exc_info.value.status_code == 400

    def test_prune_watchdog_staging_removes_expired_files(self):
        old_path = WATCHDOG_STAGING_DIR / "old-stage.csv"
        new_path = WATCHDOG_STAGING_DIR / "new-stage.csv"
        old_path.write_text("a,b\n1,2\n", encoding="utf-8")
        new_path.write_text("a,b\n1,2\n", encoding="utf-8")
        try:
            stale_time = (datetime.now(timezone.utc) - timedelta(days=2)).timestamp()
            old_path.touch()
            new_path.touch()
            import os
            os.utime(old_path, (stale_time, stale_time))

            _prune_watchdog_staging()

            assert not old_path.exists()
            assert new_path.exists()
        finally:
            if old_path.exists():
                old_path.unlink()
            if new_path.exists():
                new_path.unlink()

    def test_trigger_pipeline_from_path_preserves_original_filename(self, monkeypatch, sample_csv_content):
        from app.api import routes

        raw_path = RAW_DIR / "OriginalDirty.csv"
        raw_path.write_text(sample_csv_content, encoding="utf-8")
        captured: dict[str, object] = {}

        async def fake_run_pipeline_job(source_path, batch_id, progress_cb, cleanup_path=None):
            captured["source_path"] = source_path
            captured["cleanup_path"] = cleanup_path

        async def fake_broadcast_global(message):
            captured["message"] = message

        monkeypatch.setattr(routes, "_run_pipeline_job", fake_run_pipeline_job)
        monkeypatch.setattr(routes.manager, "broadcast_global", fake_broadcast_global)

        try:
            batch_id = asyncio.run(routes.trigger_pipeline_from_path(str(raw_path)))
            asyncio.run(asyncio.sleep(0))

            db = SessionLocal()
            try:
                run = db.query(PipelineRun).filter(PipelineRun.batch_id == batch_id).first()
                assert run is not None
                assert run.source_file == "OriginalDirty.csv"
            finally:
                db.close()

            assert captured["message"]["payload"]["data"]["file"] == "OriginalDirty.csv"
            assert Path(captured["source_path"]).name != "OriginalDirty.csv"
        finally:
            if raw_path.exists():
                raw_path.unlink()
            cleanup_path = captured.get("cleanup_path")
            if cleanup_path and Path(cleanup_path).exists():
                Path(cleanup_path).unlink()


class TestFilenameSecurity:
    """Verify uploaded filenames are sanitized."""

    def test_safe_filename_strips_paths_and_hidden_prefixes(self):
        filename = _safe_filename(r"..\..\secret\.env")
        assert ".." not in filename
        assert "\\" not in filename
        assert "/" not in filename
        assert ":" not in filename
        assert filename.endswith("env")
