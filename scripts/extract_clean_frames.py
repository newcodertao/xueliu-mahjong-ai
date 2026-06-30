from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2


IMAGE_QUALITY = 92
VIDEO_SUFFIXES = {".mp4", ".ts", ".mov", ".avi", ".mkv"}
CLASSES = [f"{rank}{suit}" for suit in ("W", "T", "B") for rank in range(1, 10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract clean manual-label train/test frames.")
    parser.add_argument("--video-root", default=r"G:\video")
    parser.add_argument("--real-test-video", default=r"G:\video\2026-06-29 15-42-34.mp4")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--test-output", required=True)
    parser.add_argument("--train-count", type=int, default=400)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--start-margin-seconds", type=float, default=5.0)
    parser.add_argument("--end-margin-seconds", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_root = Path(args.video_root)
    real_test_video = Path(args.real_test_video)
    train_output = Path(args.train_output)
    test_output = Path(args.test_output)

    if not video_root.exists():
        raise FileNotFoundError(video_root)
    if not real_test_video.exists():
        raise FileNotFoundError(real_test_video)

    all_videos = sorted(
        path for path in video_root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    train_videos = [path for path in all_videos if path.resolve() != real_test_video.resolve()]
    if not train_videos:
        raise RuntimeError("No training videos found after excluding the real test video.")

    prepare_output(train_output, args.overwrite)
    prepare_output(test_output, args.overwrite)

    test_infos = [probe_video(real_test_video)]
    train_infos = [info for video in train_videos if (info := probe_video(video)).opened and info.duration > 0]
    if not test_infos[0].opened:
        raise RuntimeError(f"Cannot open test video: {real_test_video}")

    test_rows = extract_from_videos(
        test_infos,
        test_output,
        args.test_count,
        prefix="gold",
        start_margin=args.start_margin_seconds,
        end_margin=args.end_margin_seconds,
    )
    train_rows = extract_from_videos(
        train_infos,
        train_output,
        args.train_count,
        prefix="train",
        start_margin=args.start_margin_seconds,
        end_margin=args.end_margin_seconds,
    )

    write_common_files(train_output, train_rows, train_infos, "Clean manual training candidates.")
    write_common_files(test_output, test_rows, test_infos, "Clean gold test candidates; never train on these.")

    print(f"test_images={len(test_rows)} output={test_output}")
    print(f"train_images={len(train_rows)} output={train_output}")
    print("train_videos:")
    for info in train_infos:
        print(f"- {info.path} duration={info.duration:.1f}s frames={info.frames} fps={info.fps:.3f}")
    print(f"test_video={test_infos[0].path} duration={test_infos[0].duration:.1f}s")


class VideoInfo:
    def __init__(self, path: Path, opened: bool, fps: float, frames: int) -> None:
        self.path = path
        self.opened = opened
        self.fps = fps
        self.frames = frames
        self.duration = frames / fps if fps else 0.0


def probe_video(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    opened = capture.isOpened()
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    return VideoInfo(path=path, opened=opened, fps=fps, frames=frames)


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def allocate_counts(infos: list[VideoInfo], total: int) -> list[int]:
    durations = [max(0.0, info.duration) for info in infos]
    duration_sum = sum(durations)
    if duration_sum <= 0:
        base = total // len(infos)
        allocations = [base for _ in infos]
        for i in range(total - sum(allocations)):
            allocations[i % len(allocations)] += 1
        return allocations

    exact = [total * duration / duration_sum for duration in durations]
    allocations = [int(value) for value in exact]
    remainders = sorted(((value - int(value), index) for index, value in enumerate(exact)), reverse=True)
    for _fraction, index in remainders[: total - sum(allocations)]:
        allocations[index] += 1
    return allocations


def extract_from_videos(
    infos: list[VideoInfo],
    output: Path,
    total_count: int,
    prefix: str,
    start_margin: float,
    end_margin: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    allocations = allocate_counts(infos, total_count)
    for video_index, (info, count) in enumerate(zip(infos, allocations), start=1):
        if count <= 0:
            continue
        capture = cv2.VideoCapture(str(info.path), cv2.CAP_FFMPEG)
        usable_start = min(start_margin, max(0.0, info.duration))
        usable_end = max(usable_start, info.duration - end_margin)
        usable_duration = max(0.0, usable_end - usable_start)
        for local_index in range(1, count + 1):
            second = usable_start + usable_duration * local_index / (count + 1)
            frame_index = int(round(second * info.fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            image_name = f"{prefix}_v{video_index:02d}_{local_index:04d}_{second:08.2f}s.jpg"
            image_path = output / image_name
            encoded_ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), IMAGE_QUALITY])
            if not encoded_ok:
                continue
            image_path.write_bytes(buffer.tobytes())
            rows.append(
                {
                    "image": image_name,
                    "video_index": video_index,
                    "source_video": str(info.path),
                    "seconds": f"{second:.2f}",
                    "frame_index": frame_index,
                    "fps": f"{info.fps:.3f}",
                }
            )
        capture.release()
    return rows


def write_common_files(output: Path, rows: list[dict[str, object]], infos: list[VideoInfo], note: str) -> None:
    (output / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    with (output / "source_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "video_index", "source_video", "seconds", "frame_index", "fps"],
        )
        writer.writeheader()
        writer.writerows(rows)
    (output / "README.txt").write_text(
        f"{note}\n"
        "These images are for pure manual labeling. Do not auto-label them as training truth.\n"
        f"images={len(rows)}\n"
        "videos:\n"
        + "\n".join(
            f"- index={index} path={info.path} duration={info.duration:.2f}s fps={info.fps:.3f} frames={info.frames}"
            for index, info in enumerate(infos, start=1)
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
