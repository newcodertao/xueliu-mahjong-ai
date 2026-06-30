from __future__ import annotations

from ultralytics import YOLO


def main() -> None:
    model = YOLO(r"F:\xueliu-mahjong-ai\models\yolo\xueliu_tiles_rotated_plus_human_round1_640.pt")
    model.train(
        data=r"F:\xueliu-mahjong-ai\datasets\xueliu_tiles_rotated_plus_human_round1\data.yaml",
        epochs=8,
        imgsz=960,
        batch=12,
        device=0,
        workers=4,
        patience=4,
        project=r"F:\xueliu-mahjong-ai\runs\detect",
        name="rotated_plus_human_round1_highres",
        exist_ok=True,
        pretrained=True,
        cache=False,
        verbose=True,
    )


if __name__ == "__main__":
    main()
