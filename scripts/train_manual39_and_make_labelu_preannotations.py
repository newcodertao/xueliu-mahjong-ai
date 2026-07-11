from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


PROJECT = Path(r"F:\xueliu-mahjong-ai")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a clean YOLO model from manual LabelU labels and create LabelU pre-annotations."
    )
    parser.add_argument(
        "--zip",
        default=str(
            PROJECT
            / "data"
            / "clean_start"
            / "train_manual_set"
            / "train_manual_set_clean_v1.zip"
        ),
    )
    parser.add_argument(
        "--images",
        default=str(
            PROJECT
            / "data"
            / "clean_start"
            / "train_manual_set"
            / "labelu_upload_images"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=str(PROJECT / "datasets" / "xueliu_manual39_clean_v1"),
    )
    parser.add_argument(
        "--base-model",
        default=str(PROJECT / "yolo11n.pt"),
        help="Use a clean pretrained base, not a possibly polluted previous mahjong model.",
    )
    parser.add_argument(
        "--output-model",
        default=str(PROJECT / "models" / "yolo" / "xueliu_manual39_clean_v1.pt"),
    )
    parser.add_argument(
        "--prelabel-output",
        default=str(
            PROJECT
            / "data"
            / "clean_start"
            / "train_manual_set"
            / "manual39_clean_v1_labelu_preannotations"
        ),
    )
    parser.add_argument("--run-name", default="manual39_clean_v1_high_precision")
    parser.add_argument("--prelabel-prefix", default="labelu_preannotations_manual39_clean_v1")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.55)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse --output-model and only regenerate LabelU pre-annotation files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip)
    image_dir = Path(args.images)
    dataset = Path(args.dataset)
    base_model = Path(args.base_model)
    output_model = Path(args.output_model)
    prelabel_output = Path(args.prelabel_output)

    class_names, label_members = read_zip_labels(zip_path)
    manual_stems = sorted(path.stem for path in label_members)
    if args.skip_train:
        if not output_model.exists():
            raise FileNotFoundError(output_model)
    else:
        build_dataset(zip_path, image_dir, dataset, class_names, manual_stems)
        train_model(dataset, base_model, args)

        best = PROJECT / "runs" / "detect" / args.run_name / "weights" / "best.pt"
        if not best.exists():
            raise FileNotFoundError(best)
        output_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, output_model)

    remaining_images = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES and path.stem not in set(manual_stems)
    )
    image_ids = {
        path.name: index
        for index, path in enumerate(
            sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
            start=1,
        )
    }
    make_preannotations(
        model_path=output_model,
        images=remaining_images,
        image_ids=image_ids,
        output=prelabel_output,
        class_names=class_names,
        prelabel_prefix=args.prelabel_prefix,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )
    write_round_manifest(args, dataset, output_model, class_names, len(manual_stems), len(remaining_images))

    print(f"manual_images={len(manual_stems)}")
    print(f"remaining_images={len(remaining_images)}")
    print(f"dataset={dataset}")
    print(f"model={output_model}")
    print(f"preannotations={prelabel_output / (args.prelabel_prefix + '_legacy_upload.json')}")


def read_zip_labels(zip_path: Path) -> tuple[list[str], list[Path]]:
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        class_names = archive.read("classes.txt").decode("utf-8-sig").splitlines()
        class_names = [name.strip() for name in class_names if name.strip()]
        label_members = [
            Path(name)
            for name in archive.namelist()
            if name.lower().endswith(".txt") and Path(name).name != "classes.txt"
        ]
    if not class_names:
        raise ValueError("classes.txt is empty")
    if not label_members:
        raise ValueError("No YOLO label files in zip")
    return class_names, label_members


def build_dataset(
    zip_path: Path,
    image_dir: Path,
    dataset: Path,
    class_names: list[str],
    manual_stems: list[str],
) -> None:
    if dataset.exists():
        shutil.rmtree(dataset)
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(20260701)
    stems = manual_stems[:]
    rng.shuffle(stems)
    val_count = max(4, round(len(stems) * 0.15)) if len(stems) >= 20 else max(1, len(stems) // 5)
    val_stems = set(stems[:val_count])

    stats = {"train_images": 0, "val_images": 0, "boxes": 0, "missing_images": []}
    with zipfile.ZipFile(zip_path) as archive:
        for stem in stems:
            image_path = find_image(image_dir, stem)
            if image_path is None:
                stats["missing_images"].append(stem)
                continue
            split = "val" if stem in val_stems else "train"
            label_name = f"{stem}.txt"
            label_text = archive.read(label_name).decode("utf-8-sig")
            boxes = [line for line in label_text.splitlines() if line.strip()]
            stats["boxes"] += len(boxes)
            shutil.copy2(image_path, dataset / "images" / split / image_path.name)
            (dataset / "labels" / split / label_name).write_text(
                "\n".join(boxes) + ("\n" if boxes else ""),
                encoding="utf-8",
            )
            stats[f"{split}_images"] += 1

    (dataset / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    (dataset / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(dataset).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "names": {index: name for index, name in enumerate(class_names)},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (dataset / "build_summary.yaml").write_text(
        yaml.safe_dump(stats, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def find_image(directory: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def train_model(dataset: Path, base_model: Path, args: argparse.Namespace) -> None:
    if not base_model.exists():
        raise FileNotFoundError(base_model)
    model = YOLO(str(base_model))
    model.train(
        data=str(dataset / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,
        patience=35,
        project=str(PROJECT / "runs" / "detect"),
        name=args.run_name,
        exist_ok=True,
        pretrained=True,
        cache=False,
        cos_lr=True,
        close_mosaic=25,
        hsv_h=0.01,
        hsv_s=0.25,
        hsv_v=0.25,
        degrees=3.0,
        translate=0.05,
        scale=0.25,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.35,
        mixup=0.0,
        copy_paste=0.0,
        verbose=True,
    )


def make_preannotations(
    model_path: Path,
    images: list[Path],
    image_ids: dict[str, int],
    output: Path,
    class_names: list[str],
    prelabel_prefix: str,
    imgsz: int,
    conf: float,
    iou: float,
) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    preview_dir = output / "preview_top_left_label"
    preview_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    jsonl_path = output / f"{prelabel_prefix}.jsonl"
    json_path = output / f"{prelabel_prefix}.json"
    legacy_json_path = output / f"{prelabel_prefix}_legacy_upload.json"
    summary_rows = []
    records = []
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for image_path in images:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            detections = predict(model, image_path, width, height, imgsz, conf, iou)
            record = labelu_record(image_ids[image_path.name], image_path.name, width, height, detections)
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            cv2.imwrite(str(preview_dir / image_path.name), draw_preview(image, detections))
            summary_rows.append(
                {
                    "image": image_path.name,
                    "detections": len(detections),
                    "high_conf_075": sum(1 for det in detections if det["score"] >= 0.75),
                }
            )

    # LabelU supports jsonl directly. Its .json importer expects a legacy wrapper
    # where result is itself a JSON string, so write both variants explicitly.
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    legacy_records = [labelu_legacy_json_record(record) for record in records]
    legacy_json_path.write_text(json.dumps(legacy_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    write_summary_csv(output / "preannotation_summary.csv", summary_rows)
    (output / "README.txt").write_text(
        "LabelU pre-annotations generated from a clean manual39 model.\n"
        f"Upload {legacy_json_path.name} as the pre-annotation file in LabelU.\n"
        f"{json_path.name} is also valid JSON for inspection; {jsonl_path.name} is not needed for this workflow.\n"
        f"model={model_path}\n"
        f"images={len(summary_rows)}\n"
        f"conf={conf}\n"
        f"iou={iou}\n"
        f"imgsz={imgsz}\n",
        encoding="utf-8",
    )


def predict(model: YOLO, image_path: Path, width: int, height: int, imgsz: int, conf: float, iou: float):
    results = model.predict(source=str(image_path), imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        names = result.names
        for box in boxes:
            label = str(names[int(box.cls[0].item())]).strip()
            score = float(box.conf[0].item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            x1 = max(0.0, min(x1, width))
            x2 = max(0.0, min(x2, width))
            y1 = max(0.0, min(y1, height))
            y2 = max(0.0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({"label": label, "score": score, "xyxy": (x1, y1, x2, y2)})
    return detections


def labelu_record(data_id: int, image_name: str, width: int, height: int, detections: list[dict]) -> dict:
    rect_annotation = {
        "toolName": "rectTool",
        "result": [labelu_rect(det, index + 1) for index, det in enumerate(detections)],
    }
    return {
        "id": data_id,
        "sample_name": image_name,
        "config": {"rectTool": labelu_label_config()},
        "meta_data": {"width": width, "height": height, "rotate": 0},
        "annotations": {"rectTool": rect_annotation},
    }


def labelu_legacy_json_record(record: dict) -> dict:
    result = {
        "width": record["meta_data"]["width"],
        "height": record["meta_data"]["height"],
        "rotate": record["meta_data"]["rotate"],
        "annotations": [record["annotations"]["rectTool"]],
    }
    return {
        "id": record["id"],
        "result": json.dumps(result, ensure_ascii=False),
        "url": "",
        "folder": "",
        "fileName": record["sample_name"],
        "meta": {
            "source_type": "ai_generated",
            "provider": "local_yolo_manual39_clean",
            "model": "xueliu_manual39_clean_v1",
            "warning_message": "AI prelabel; manual review required before training.",
        },
    }


def labelu_rect(det: dict, order: int) -> dict:
    x1, y1, x2, y2 = det["xyxy"]
    return {
        "id": f"manual39-{order:04d}",
        "order": order,
        "label": det["label"],
        "visible": True,
        "x": x1,
        "y": y1,
        "width": x2 - x1,
        "height": y2 - y1,
        "attributes": {"score": f"{det['score']:.4f}"},
    }


def labelu_label_config() -> list[dict]:
    colors = [
        "#ff7a00",
        "#ff4d4f",
        "#40a9ff",
        "#faad14",
        "#2dbf44",
        "#ff713d",
        "#4d5bff",
        "#c94ddd",
        "#f7598f",
        "#d48806",
        "#eb2f96",
        "#f2c94c",
        "#d6e64f",
        "#8b6cff",
        "#95de64",
        "#36cfc9",
        "#40a9cc",
        "#91d5ff",
        "#9c9ede",
        "#b37feb",
        "#ad8b00",
        "#ff4d4f",
        "#69c0ff",
        "#faad14",
        "#52c41a",
        "#ff7a45",
        "#597ef7",
    ]
    names = [f"{rank}{suit}" for suit in ("W", "B", "T") for rank in range(1, 10)]
    return [
        {
            "key": name,
            "value": name,
            "color": colors[index % len(colors)],
        }
        for index, name in enumerate(names)
    ]


def draw_preview(image, detections: list[dict]):
    canvas = image.copy()
    colors = {"W": (0, 220, 255), "T": (255, 140, 0), "B": (0, 220, 0)}
    for det in detections:
        label = det["label"]
        x1, y1, x2, y2 = map(int, det["xyxy"])
        color = colors.get(label[-1], (0, 255, 0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return canvas


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    lines = ["image,detections,high_conf_075"]
    for row in rows:
        lines.append(f"{row['image']},{row['detections']},{row['high_conf_075']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_round_manifest(
    args: argparse.Namespace,
    dataset: Path,
    output_model: Path,
    class_names: list[str],
    manual_count: int,
    remaining_count: int,
) -> None:
    manifest = PROJECT / "models" / "yolo" / "training_rounds.yaml"
    existing = yaml.safe_load(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    rounds = existing.get("rounds", []) if isinstance(existing, dict) else []
    rounds.append(
        {
            "name": "manual39_clean_v1",
            "model": str(output_model),
            "previous_model": str(Path(args.base_model)),
            "run_dir": str(PROJECT / "runs" / "detect" / args.run_name),
            "dataset": str(dataset),
            "manual_images": manual_count,
            "prelabeled_remaining_images": remaining_count,
            "classes": class_names,
            "rollback_note": "This round starts from a clean YOLO base and uses only 39 manually verified images.",
        }
    )
    manifest.write_text(
        yaml.safe_dump({"rounds": rounds}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
