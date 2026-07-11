from __future__ import annotations

import argparse
import json
from pathlib import Path

from xueliu_ai.capture.collector import collect_frames
from xueliu_ai.capture.roi_calibrator import calibrate_roi
from xueliu_ai.capture.roi_config import Roi, update_roi
from xueliu_ai.dataset.dataset_builder import build_yolo_dataset
from xueliu_ai.dataset.dedupe import dedupe_images
from xueliu_ai.dataset.roboflow_downloader import download_roboflow_dataset
from xueliu_ai.dataset.roboflow_mahjong_converter import convert_roboflow_mahjong_to_xueliu
from xueliu_ai.dataset.sample_manifest import build_sample_manifest
from xueliu_ai.dataset.video_pseudo_labeler import (
    extract_video_frames,
    probe_video,
    pseudo_label_frames,
)
from xueliu_ai.evaluation.replay_test import run_replay_test
from xueliu_ai.evaluation.video_replay import replay_video
from xueliu_ai.game_logging.game_logger import GameLogger
from xueliu_ai.game_logging.review_report import generate_markdown_report, summarize_jsonl
from xueliu_ai.mahjong.tiles import parse_tiles
from xueliu_ai.realtime import run_realtime_loop
from xueliu_ai.realtime_table import run_realtime_table_loop
from xueliu_ai.strategy.discard_advisor import advise_discard
from xueliu_ai.ui.debug_viewer import show_debug_image
from xueliu_ai.ui.realtime_app import launch_realtime_app
from xueliu_ai.vision.benchmark import benchmark_hand_folder
from xueliu_ai.vision.detection_exporter import copy_hard_case, export_detection_result
from xueliu_ai.vision.detection_validator import validate_hand_detections
from xueliu_ai.vision.yolo_detector import YoloDetector


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="xueliu-ai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Capture fullscreen and configured hand ROI images.")
    collect.add_argument("--interval", type=float, default=0.5)
    collect.add_argument("--limit", type=int)

    roi = subparsers.add_parser("roi-calibrate", help="Interactively select and save an ROI.")
    roi.add_argument("--name", default="my_hand")
    roi.add_argument("--monitor", type=int, default=1)

    roi_set = subparsers.add_parser("roi-set", help="Set an ROI from explicit coordinates.")
    roi_set.add_argument("--name", default="my_hand")
    roi_set.add_argument("--x", type=int, required=True)
    roi_set.add_argument("--y", type=int, required=True)
    roi_set.add_argument("--width", type=int, required=True)
    roi_set.add_argument("--height", type=int, required=True)

    dataset = subparsers.add_parser("build-dataset", help="Create YOLO dataset directories and data.yaml.")
    dataset.add_argument("--dir", default="datasets/xueliu_tiles_v1")

    manifest = subparsers.add_parser("manifest", help="Build a CSV manifest for captured samples.")
    manifest.add_argument("--raw-dir", default="data/raw")
    manifest.add_argument("--output", default="data/sample_manifest.csv")

    dedupe = subparsers.add_parser("dedupe-samples", help="Copy visually unique samples to a curated folder.")
    dedupe.add_argument("--source", default="data/raw/my_hand")
    dedupe.add_argument("--output", default="data/curated/my_hand_unique")
    dedupe.add_argument("--max-distance", type=int, default=8)

    roboflow = subparsers.add_parser("download-roboflow", help="Download a Roboflow dataset using ROBOFLOW_API_KEY from .env.")
    roboflow.add_argument("--workspace", required=True)
    roboflow.add_argument("--project", required=True)
    roboflow.add_argument("--version", type=int, required=True)
    roboflow.add_argument("--format", default="yolov11")
    roboflow.add_argument("--output", default="external_datasets/roboflow")
    roboflow.add_argument("--no-overwrite", action="store_true")

    convert_rf = subparsers.add_parser("convert-roboflow-mahjong", help="Convert Roboflow 42-class Mahjong data to 27 Xueliu classes.")
    convert_rf.add_argument("--source", default="external_datasets/roboflow_mahjong")
    convert_rf.add_argument("--output", default="datasets/xueliu_tiles_roboflow_v1")

    video_probe = subparsers.add_parser("video-probe", help="Read basic metadata from a video file.")
    video_probe.add_argument("--video", required=True)

    video_frames = subparsers.add_parser("video-extract", help="Extract spaced frames from a Mahjong video.")
    video_frames.add_argument("--video", required=True)
    video_frames.add_argument("--output", default="data/video_frames")
    video_frames.add_argument("--every-seconds", type=float, default=3.0)
    video_frames.add_argument("--start-seconds", type=float, default=0.0)
    video_frames.add_argument("--end-seconds", type=float)
    video_frames.add_argument("--max-frames", type=int)

    video_pseudo = subparsers.add_parser("video-pseudo-label", help="Create a YOLO dataset from video frames using high-confidence model predictions.")
    video_pseudo.add_argument("--model", required=True)
    video_pseudo.add_argument("--image-dir", required=True)
    video_pseudo.add_argument("--output", default="datasets/xueliu_tiles_video_pseudo_v1")
    video_pseudo.add_argument("--conf", type=float, default=0.85)
    video_pseudo.add_argument("--iou", type=float, default=0.5)
    video_pseudo.add_argument("--min-detections", type=int, default=5)
    video_pseudo.add_argument("--val-ratio", type=float, default=0.15)

    detect = subparsers.add_parser("detect", help="Run YOLO detection on an image or folder.")
    detect.add_argument("--model", required=True)
    detect.add_argument("--source", required=True)
    detect.add_argument("--conf", type=float, default=0.6)
    detect.add_argument("--iou", type=float, default=0.5)
    detect.add_argument("--export", action="store_true")
    detect.add_argument("--hard-case-invalid", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="Benchmark hand recognition on labeled samples.")
    benchmark.add_argument("--model", required=True)
    benchmark.add_argument("--image-dir", default="data/raw/my_hand")
    benchmark.add_argument("--label-dir", default="data/labeled/my_hand")
    benchmark.add_argument("--output", default="data/reviews/benchmark.json")
    benchmark.add_argument("--conf", type=float, default=0.6)
    benchmark.add_argument("--iou", type=float, default=0.5)

    advise = subparsers.add_parser("advise", help="Recommend a discard for a 14-tile hand.")
    advise.add_argument("--hand", required=True, help="Comma-separated tiles, e.g. 1W,2W,3W")
    advise.add_argument("--missing-suit", choices=["W", "T", "B", "w", "t", "b"])
    advise.add_argument("--log", default=None)

    debug = subparsers.add_parser("debug-viewer", help="Show image with configured ROI overlays.")
    debug.add_argument("--image", required=True)

    realtime = subparsers.add_parser("realtime", help="Run realtime screenshot, recognition and advice loop.")
    realtime.add_argument("--model", required=True)
    realtime.add_argument("--missing-suit", choices=["W", "T", "B", "w", "t", "b"])
    realtime.add_argument("--interval", type=float, default=0.5)
    realtime.add_argument("--limit", type=int)

    realtime_table = subparsers.add_parser(
        "realtime-table",
        help="Run realtime full-table screenshot recognition, zone parsing and advice loop.",
    )
    realtime_table.add_argument(
        "--model",
        default="models/yolo/xueliu_final325_plus_longjing39_plus83_clean_v1_0709.pt",
    )
    realtime_table.add_argument("--missing-suit", choices=["W", "T", "B", "w", "t", "b"])
    realtime_table.add_argument("--interval", type=float, default=0.25)
    realtime_table.add_argument("--limit", type=int)
    realtime_table.add_argument("--roi-name", default="table")
    realtime_table.add_argument("--conf", type=float, default=0.75)
    realtime_table.add_argument("--iou", type=float, default=0.45)
    realtime_table.add_argument("--imgsz", type=int, default=1280)
    realtime_table.add_argument("--no-show", action="store_true")
    realtime_table.add_argument("--save-preview-dir")
    realtime_table.add_argument("--log", default="data/games/realtime_table.jsonl")

    video_replay = subparsers.add_parser("video-replay", help="Replay a video through the table recognizer and save diagnostics.")
    video_replay.add_argument("--video", required=True)
    video_replay.add_argument("--model", default="models/yolo/xueliu_final325_plus_longjing39_plus83_clean_v1_0709.pt")
    video_replay.add_argument("--output", default="data/replays/latest")
    video_replay.add_argument("--every-seconds", type=float, default=1.0)
    video_replay.add_argument("--max-frames", type=int, default=120)
    video_replay.add_argument("--conf", type=float, default=0.75)
    video_replay.add_argument("--iou", type=float, default=0.45)
    video_replay.add_argument("--imgsz", type=int, default=1280)
    video_replay.add_argument("--roi-name", default="table")
    video_replay.add_argument("--no-images", action="store_true")

    replay_test = subparsers.add_parser("replay-test", help="Run curated gold-frame regression checks.")
    replay_test.add_argument("--gold", default="data/gold_replay/gold_cases.json")
    replay_test.add_argument("--model", default="models/yolo/xueliu_final325_plus_longjing39_plus83_clean_v1_0709.pt")
    replay_test.add_argument("--output")
    replay_test.add_argument("--conf", type=float, default=0.75)
    replay_test.add_argument("--iou", type=float, default=0.45)
    replay_test.add_argument("--imgsz", type=int, default=1280)
    replay_test.add_argument("--no-images", action="store_true")

    subparsers.add_parser("realtime-ui", help="Launch the realtime recognition and advice UI.")

    review = subparsers.add_parser("review", help="Summarize a JSONL game log.")
    review.add_argument("--log", default="data/games/session.jsonl")

    report = subparsers.add_parser("report", help="Generate a markdown review report from JSONL logs.")
    report.add_argument("--log", default="data/games/session.jsonl")
    report.add_argument("--output", default="data/reviews/report.md")

    args = parser.parse_args(argv)

    if args.command == "collect":
        count = collect_frames(interval_seconds=args.interval, limit=args.limit)
        print(f"saved {count} frame(s)")
    elif args.command == "roi-calibrate":
        selected = calibrate_roi(name=args.name, monitor=args.monitor)
        print(json.dumps(selected.to_dict(), ensure_ascii=False))
    elif args.command == "roi-set":
        selected = Roi(x=args.x, y=args.y, width=args.width, height=args.height)
        update_roi(args.name, selected)
        print(json.dumps(selected.to_dict(), ensure_ascii=False))
    elif args.command == "build-dataset":
        root = build_yolo_dataset(args.dir)
        print(f"dataset ready: {root}")
    elif args.command == "manifest":
        output = build_sample_manifest(args.raw_dir, args.output)
        print(f"manifest ready: {output}")
    elif args.command == "dedupe-samples":
        result = dedupe_images(args.source, args.output, args.max_distance)
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    elif args.command == "download-roboflow":
        result = download_roboflow_dataset(
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            fmt=args.format,
            output_dir=args.output,
            overwrite=not args.no_overwrite,
        )
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    elif args.command == "convert-roboflow-mahjong":
        result = convert_roboflow_mahjong_to_xueliu(args.source, args.output)
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    elif args.command == "video-probe":
        result = probe_video(args.video)
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    elif args.command == "video-extract":
        result = extract_video_frames(
            video_path=args.video,
            output_dir=args.output,
            every_seconds=args.every_seconds,
            start_seconds=args.start_seconds,
            end_seconds=args.end_seconds,
            max_frames=args.max_frames,
        )
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    elif args.command == "video-pseudo-label":
        result = pseudo_label_frames(
            model_path=args.model,
            image_dir=args.image_dir,
            dataset_dir=args.output,
            conf=args.conf,
            iou=args.iou,
            min_detections=args.min_detections,
            val_ratio=args.val_ratio,
        )
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    elif args.command == "detect":
        detector = YoloDetector(args.model)
        detections = detector.detect_image(Path(args.source), conf=args.conf, iou=args.iou)
        result = validate_hand_detections(detections, args.conf, args.iou)
        if args.export:
            export_detection_result(args.source, detections, result.valid, result.reason)
        if args.hard_case_invalid and not result.valid:
            copy_hard_case(args.source, reason=result.reason)
        print(
            json.dumps(
                {
                    "valid": result.valid,
                    "tiles": result.tiles,
                    "reason": result.reason,
                    "detections": [det.to_dict() for det in detections],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "benchmark":
        summary = benchmark_hand_folder(
            model_path=args.model,
            image_dir=args.image_dir,
            label_dir=args.label_dir,
            conf=args.conf,
            iou=args.iou,
            output=args.output,
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "advise":
        tiles = parse_tiles(args.hand)
        advice = advise_discard(tiles, args.missing_suit)
        payload = {
            "recommended": advice.recommended,
            "explanation": advice.explanation,
            "candidates": [candidate.__dict__ for candidate in advice.candidates],
        }
        if args.log:
            GameLogger(args.log).log("advice", {"hand": tiles, **payload})
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "debug-viewer":
        show_debug_image(args.image)
    elif args.command == "realtime":
        run_realtime_loop(
            model_path=args.model,
            missing_suit=args.missing_suit,
            interval_seconds=args.interval,
            limit=args.limit,
        )
    elif args.command == "realtime-table":
        run_realtime_table_loop(
            model_path=args.model,
            missing_suit=args.missing_suit,
            interval_seconds=args.interval,
            limit=args.limit,
            roi_name=args.roi_name,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            show=not args.no_show,
            save_preview_dir=args.save_preview_dir,
            log_path=args.log,
        )
    elif args.command == "video-replay":
        summary = replay_video(
            video_path=args.video,
            model_path=args.model,
            output_dir=args.output,
            every_seconds=args.every_seconds,
            max_frames=args.max_frames,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            save_images=not args.no_images,
            roi_name=args.roi_name,
        )
        print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    elif args.command == "replay-test":
        summary = run_replay_test(
            gold_path=args.gold,
            model_path=args.model,
            output_dir=args.output,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            save_images=not args.no_images,
        )
        print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    elif args.command == "realtime-ui":
        launch_realtime_app()
    elif args.command == "review":
        print(json.dumps(summarize_jsonl(args.log), ensure_ascii=False, indent=2))
    elif args.command == "report":
        output = generate_markdown_report(args.log, args.output)
        print(f"report ready: {output}")


if __name__ == "__main__":
    main()
