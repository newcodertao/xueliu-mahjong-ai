from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT / "models" / "yolo" / "xueliu_final325_plus_longjing39_plus83_clean_v1_0709.pt"
DEFAULT_OUTPUT = PROJECT.parent / "xueliu-mahjong-ai_runtime_package_20260705.zip"

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    ".venv-labeling",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "data",
    "datasets",
    "external_datasets",
    "runs",
    "reports",
}
EXCLUDE_FILES = {".env", "IP账号密码.txt"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".mp4", ".avi", ".mov", ".mkv", ".zip"}
INCLUDE_MODEL_FILES = {
    "models/yolo/xueliu_final325_plus_longjing39_plus83_clean_v1_0709.pt",
    "models/yolo/training_rounds.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a sanitized runtime package.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    args = parser.parse_args()

    output = Path(args.output).resolve()
    model = Path(args.model).resolve()
    if not model.exists():
        raise FileNotFoundError(f"model not found: {model}")

    if output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    added = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in PROJECT.rglob("*"):
            if path.is_dir() or _should_exclude(path):
                continue
            rel = path.relative_to(PROJECT).as_posix()
            if rel.startswith("models/") and rel not in INCLUDE_MODEL_FILES:
                continue
            zf.write(path, f"xueliu-mahjong-ai/{rel}")
            added += 1

    _validate_package(output)
    size_mb = output.stat().st_size / 1024 / 1024
    print(f"runtime package ready: {output} ({size_mb:.1f} MB, {added} files)")


def _should_exclude(path: Path) -> bool:
    rel_parts = path.relative_to(PROJECT).parts
    if any(part in EXCLUDE_DIRS for part in rel_parts[:-1]):
        return True
    if path.name in EXCLUDE_FILES:
        return True
    lowered = path.name.lower()
    if any(token in lowered for token in ("账号", "密码", "secret", "token")):
        return True
    return path.suffix.lower() in EXCLUDE_SUFFIXES


def _validate_package(path: Path) -> None:
    forbidden = ("IP账号密码.txt", ".env", "/data/", "/datasets/", "/runs/", "/reports/", "/.git/", "/.venv/")
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    offenders = [name for name in names if any(item in name for item in forbidden)]
    if offenders:
        raise RuntimeError("package contains forbidden files: " + ", ".join(offenders[:10]))
    model_name = "xueliu-mahjong-ai/models/yolo/xueliu_final325_plus_longjing39_plus83_clean_v1_0709.pt"
    if model_name not in names:
        raise RuntimeError(f"package is missing model: {model_name}")


if __name__ == "__main__":
    main()

