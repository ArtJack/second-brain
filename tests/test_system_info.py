"""Formatting helpers for local machine status."""

from pathlib import Path

from secondbrain import system_info


def test_format_storage_status(monkeypatch, tmp_path):
    monkeypatch.setattr(
        system_info,
        "storage_status",
        lambda: [{"path": str(tmp_path), "total": 100, "used": 40, "free": 60, "percent_used": 0.4}],
    )

    text = system_info.format_system_status("storage")

    assert "Storage:" in text
    assert str(tmp_path) in text
    assert "60.0 B free / 100.0 B total" in text


def test_storage_status_uses_disk_usage(monkeypatch, tmp_path):
    monkeypatch.setattr(
        system_info.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"total": 100, "used": 25, "free": 75})(),
    )

    rows = system_info.storage_status([Path(tmp_path)])

    assert rows == [
        {
            "path": str(tmp_path),
            "total": 100,
            "used": 25,
            "free": 75,
            "percent_used": 0.25,
        }
    ]
