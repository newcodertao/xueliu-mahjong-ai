from pathlib import Path

from xueliu_ai.dataset.roboflow_downloader import download_roboflow_dataset


def test_download_roboflow_requires_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    try:
        download_roboflow_dataset("workspace", "project", 1, env_path=tmp_path / ".env")
    except RuntimeError as exc:
        assert "ROBOFLOW_API_KEY" in str(exc)
    else:
        raise AssertionError("expected missing API key error")
