from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from xueliu_ai.paths import resolve_path


@dataclass(frozen=True)
class RoboflowDownloadResult:
    workspace: str
    project: str
    version: int
    format: str
    location: str


def download_roboflow_dataset(
    workspace: str,
    project: str,
    version: int,
    fmt: str = "yolov11",
    output_dir: str | Path = "external_datasets/roboflow",
    overwrite: bool = True,
    env_path: str | Path = ".env",
) -> RoboflowDownloadResult:
    load_dotenv(resolve_path(env_path))
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is missing. Put it in .env or set it as an env var.")

    from roboflow import Roboflow

    root = resolve_path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    dataset = rf.workspace(workspace).project(project).version(version).download(
        fmt,
        location=str(root),
        overwrite=overwrite,
    )
    location = getattr(dataset, "location", str(root))
    return RoboflowDownloadResult(
        workspace=workspace,
        project=project,
        version=version,
        format=fmt,
        location=str(location),
    )
