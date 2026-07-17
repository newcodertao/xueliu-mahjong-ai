from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from xueliu_ai.evaluation.readiness_calibration import risk_coverage_curve, select_threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="JSONL rows with core_score and core_state_correct")
    parser.add_argument("--maximum-risk", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = select_threshold(rows, maximum_risk=args.maximum_risk)
    payload = {
        "maximum_risk": args.maximum_risk,
        "selected": asdict(selected) if selected else None,
        "curve": [asdict(point) for point in risk_coverage_curve(rows)],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
