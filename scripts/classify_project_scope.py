from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from RetailPilot.conversation_replay import load_conversations
from RetailPilot.llm_config import judge_config_status
from RetailPilot.scope_eval import (
    LLM_SCOPE_VERSION,
    build_scope_cases,
    classify_scope_batch,
    materialize_llm_scope_corpus,
)


def _chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def _batch_id(cases: list[dict[str, Any]]) -> str:
    joined = "\n".join(str(item["case_id"]) for item in cases)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _read_checkpoint(path: Path) -> tuple[dict[str, dict[str, Any]], int, int]:
    verdicts: dict[str, dict[str, Any]] = {}
    llm_batches = 0
    fallback_batches = 0
    if not path.exists():
        return verdicts, llm_batches, fallback_batches
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("classifier_mode") != "llm":
                fallback_batches += 1
                continue
            llm_batches += 1
            for item in record.get("items") or []:
                verdicts[str(item["case_id"])] = item
    return verdicts, llm_batches, fallback_batches


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _materialize(
    output: Path,
    conversations: list[dict[str, Any]],
    verdicts: dict[str, dict[str, Any]],
    *,
    run_summary: dict[str, Any],
) -> dict[str, Any]:
    eligible, filtered, uncertain, scope_summary = materialize_llm_scope_corpus(
        conversations, verdicts
    )
    summary = {**run_summary, **scope_summary}
    (output / "scope_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_jsonl(output / "eligible_conversations.jsonl", eligible)
    _write_jsonl(output / "scope_filtered_turns.jsonl", filtered)
    _write_jsonl(output / "scope_uncertain_turns.jsonl", uncertain)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch LLM project-scope classification before Agent replay"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--format", dest="input_format", choices=["jddc-uss", "jsonl"], required=True
    )
    parser.add_argument("--limit", type=int, default=3300)
    parser.add_argument("--min-user-turns", type=int, default=2)
    parser.add_argument(
        "--max-user-turns", type=int, default=0,
        help="0 means all user turns",
    )
    parser.add_argument(
        "--batch-size", type=int, default=20,
        help="Queries classified in one LLM request",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--min-batch-size", type=int, default=5,
        help="Failed batches are recursively split, never below this size",
    )
    parser.add_argument(
        "--max-batches", type=int, default=0,
        help="Debug/smoke limit; 0 means all pending batches",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=args.resume)
    checkpoint = output / "scope_batch_results.jsonl"
    conversations = load_conversations(
        args.input,
        input_format=args.input_format,
        limit=max(1, args.limit),
        min_user_turns=max(1, args.min_user_turns),
    )
    max_user_turns = None if args.max_user_turns <= 0 else args.max_user_turns
    cases = build_scope_cases(conversations, max_user_turns=max_user_turns)
    verdicts, previous_llm_batches, previous_fallback_batches = _read_checkpoint(
        checkpoint if args.resume else Path("__missing_scope_checkpoint__")
    )
    pending = [item for item in cases if str(item["case_id"]) not in verdicts]
    batches = _chunks(pending, max(1, args.batch_size))
    if args.max_batches > 0:
        batches = batches[:args.max_batches]

    mode = "a" if args.resume and checkpoint.exists() else "w"
    new_llm_batches = 0
    new_fallback_batches = 0

    def classify(cases_in_batch: list[dict[str, Any]]) -> dict[str, Any]:
        result = classify_scope_batch(cases_in_batch)
        return {
            "batch_id": _batch_id(cases_in_batch),
            "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "batch_size": len(cases_in_batch),
            **result,
        }

    min_batch_size = max(2, min(args.min_batch_size, max(1, args.batch_size)))
    with checkpoint.open(mode, encoding="utf-8") as stream:
        pending_batches = list(batches)
        while pending_batches:
            current_wave = pending_batches
            pending_batches = []
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                futures = {
                    executor.submit(classify, batch): batch for batch in current_wave
                }
                for future in as_completed(futures):
                    batch = futures[future]
                    record = future.result()
                    if record.get("classifier_mode") == "llm":
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        new_llm_batches += 1
                        for item in record.get("items") or []:
                            verdicts[str(item["case_id"])] = item
                        continue
                    if len(batch) > min_batch_size:
                        midpoint = max(min_batch_size, len(batch) // 2)
                        pending_batches.extend(
                            part for part in (batch[:midpoint], batch[midpoint:]) if part
                        )
                        continue
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    new_fallback_batches += 1

    resolved = len(verdicts)
    unresolved = len(cases) - resolved
    run_summary = {
        "scope_gate_version": LLM_SCOPE_VERSION,
        "source": str(args.input.resolve()),
        "input_format": args.input_format,
        "batch_size": max(1, args.batch_size),
        "workers": max(1, args.workers),
        "min_batch_size": min_batch_size,
        "total_cases": len(cases),
        "resolved_cases": resolved,
        "unresolved_cases": unresolved,
        "llm_request_upper_bound": (
            len(cases) + max(1, args.batch_size) - 1
        ) // max(1, args.batch_size),
        "checkpoint_llm_batches": previous_llm_batches + new_llm_batches,
        "checkpoint_fallback_batches": previous_fallback_batches + new_fallback_batches,
        "judge_configuration": judge_config_status(),
        "agent_executed": False,
        "llm_judge_executed": False,
        "scope_classifier_llm_executed": resolved > 0,
    }
    summary = _materialize(
        output, conversations, verdicts, run_summary=run_summary
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                **run_summary,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "raw_data_committed": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))
    if args.require_complete and unresolved:
        raise SystemExit(f"LLM scope classification incomplete: unresolved={unresolved}")


if __name__ == "__main__":
    main()
