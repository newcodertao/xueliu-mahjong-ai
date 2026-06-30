from __future__ import annotations

import json
from pathlib import Path

from xueliu_ai.paths import resolve_path


def summarize_jsonl(log_path: str | Path) -> dict[str, int]:
    path = resolve_path(log_path)
    summary: dict[str, int] = {}
    if not path.exists():
        return summary
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line).get("event", "unknown")
            summary[event] = summary.get(event, 0) + 1
    return summary


def generate_markdown_report(
    log_path: str | Path = "data/games/session.jsonl",
    output: str | Path = "data/reviews/report.md",
) -> Path:
    path = resolve_path(log_path)
    output_path = resolve_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(path)
    advice_rows = [row for row in rows if row.get("event") == "advice"]

    lines = [
        "# 血流成河复盘报告",
        "",
        f"- 日志文件：`{path}`",
        f"- 总事件数：{len(rows)}",
        f"- 推荐事件数：{len(advice_rows)}",
        "",
        "## 事件统计",
        "",
    ]
    summary = summarize_jsonl(path)
    if summary:
        lines.extend(f"- {event}: {count}" for event, count in sorted(summary.items()))
    else:
        lines.append("- 暂无事件")

    if advice_rows:
        lines.extend(["", "## 最近推荐", ""])
        for row in advice_rows[-10:]:
            lines.append(f"- {row.get('timestamp', '')}: 推荐 `{row.get('recommended', '')}`")
            explanation = row.get("explanation")
            if explanation:
                lines.append(f"  {explanation}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
