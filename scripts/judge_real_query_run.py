from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from RetailPilot.judging import judge_payload


def _judge_input(record: dict[str, Any]) -> dict[str, Any]:
    """Build a judge payload that cannot expose qrels or ShopSimulator Gold."""
    return {
        "query": record["query"],
        "parsed_query": record.get("parsed_query") or {},
        "products": record.get("top_products") or [],
        "retrieval_context": {
            "result_count": record.get("result_count", 0),
            "relaxations": record.get("relaxations") or [],
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Real Query LLMJudge",
        "",
        f"- Run: `{report['run_id']}`",
        f"- Selected / completed / failed: {report['selected']} / "
        f"{report['completed']} / {report['failed']}",
        f"- Overall >= 2: {report['overall_ge_2_rate']:.2%}",
        f"- Top1 relevance >= 2: {report['top1_relevance_ge_2_rate']:.2%}",
        "- Judge payload excludes acceptable_skus, hard_constraints and target rank.",
        "",
        "## Failure Type",
        "",
        "| Type | Count |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {key} | {value} |" for key, value in report["failure_type_counts"].items()
    )
    lines.extend(["", "## Samples", ""])
    for item in report["items"]:
        verdict = item.get("verdict") or {}
        lines.append(
            f"- `{item['task_id']}` overall={verdict.get('overall_score')}, "
            f"failure=`{verdict.get('failure_type')}`: {verdict.get('summary', item.get('error', ''))}"
        )
    return "\n".join(lines) + "\n"


def _report(run_id: str, selected: list[dict[str, Any]], items: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in items if item.get("verdict")]
    failure_counts = Counter(
        str(item["verdict"].get("failure_type") or "unknown") for item in completed
    )
    top1_relevant = 0
    for item in completed:
        judged_items = item["verdict"].get("items") or []
        if judged_items and int(judged_items[0].get("relevance") or 0) >= 2:
            top1_relevant += 1
    return {
        "run_id": run_id,
        "selected": len(selected),
        "completed": len(completed),
        "failed": len(items) - len(completed),
        "overall_ge_2_rate": round(
            sum(int(item["verdict"].get("overall_score") or 0) >= 2 for item in completed)
            / max(1, len(completed)),
            4,
        ),
        "top1_relevance_ge_2_rate": round(top1_relevant / max(1, len(completed)), 4),
        "failure_type_counts": dict(failure_counts.most_common()),
        "items": sorted(items, key=lambda item: item["sample_index"]),
    }


def _save(run_dir: Path, report: dict[str, Any]) -> None:
    json_path = run_dir / "llm_judge.json"
    markdown_path = run_dir / "llm_judge.md"
    json_tmp = json_path.with_suffix(".json.tmp")
    markdown_tmp = markdown_path.with_suffix(".md.tmp")
    json_tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_tmp.write_text(_markdown(report), encoding="utf-8")
    json_tmp.replace(json_path)
    markdown_tmp.replace(markdown_path)


def _judge_record(record: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = {
        "sample_index": int(record["sample_index"]),
        "task_id": record["task_id"],
        "query": record["query"],
    }
    try:
        item["verdict"] = judge_payload(_judge_input(record))
    except Exception as exc:
        item["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge non-empty strict misses without Gold leakage")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero if any selected LLM Judge request fails",
    )
    args = parser.parse_args()

    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (args.run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [
        record
        for record in records
        if record.get("target_rank") is None
        and record.get("result_count", 0) > 0
        and not record.get("error")
    ][: max(1, args.limit)]

    selected_indices = {int(record["sample_index"]) for record in selected}
    checkpoint_path = args.run_dir / "llm_judge.json"
    checkpoint_items: list[dict[str, Any]] = []
    if checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint_items = [
                item
                for item in checkpoint.get("items") or []
                if int(item.get("sample_index", -1)) in selected_indices and item.get("verdict")
            ]
        except (OSError, ValueError, TypeError):
            checkpoint_items = []
    items: list[dict[str, Any]] = checkpoint_items
    completed_indices = {int(item["sample_index"]) for item in checkpoint_items}
    pending = [
        record for record in selected if int(record["sample_index"]) not in completed_indices
    ]
    _save(args.run_dir, _report(manifest["run_id"], selected, items))
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(_judge_record, record): record for record in pending}
        for future in as_completed(futures):
            items.append(future.result())
            _save(args.run_dir, _report(manifest["run_id"], selected, items))
    report = _report(manifest["run_id"], selected, items)
    _save(args.run_dir, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_complete and (report["selected"] == 0 or report["failed"]):
        raise SystemExit(
            "LLM Judge completion requirement failed: "
            f"selected={report['selected']}, failed={report['failed']}"
        )


if __name__ == "__main__":
    main()
