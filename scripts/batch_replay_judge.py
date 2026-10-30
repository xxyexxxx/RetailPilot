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

from RetailPilot.conversation_replay import llm_judge_evidence
from RetailPilot.customer_eval import (
    _CUSTOMER_JUDGE_SYSTEM,
    customer_batch_judge_prompt,
    customer_judge_evidence,
    validate_customer_batch_judge_output,
)
from RetailPilot.llm_config import invoke_judge_stream, judge_config_status
from scripts.run_multiturn_replay import write_replay_reports


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def _atomic_write(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _batch_id(cases: list[dict[str, Any]]) -> str:
    joined = "\n".join(str(item["case_id"]) for item in cases)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def build_judge_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for conversation in records:
        conversation_id = str(conversation["conversation_id"])
        history: list[dict[str, str]] = []
        for turn in conversation.get("turns") or []:
            turn_index = int(turn.get("turn_index") or 0)
            if (turn.get("scope_gate") or {}).get("in_scope") is False:
                continue
            result = {
                "reply": str(turn.get("reply") or ""),
                "intents": turn.get("intents") or [],
                "task_results": turn.get("task_results") or [],
                "trace": turn.get("trace") or [],
                "grounding": turn.get("grounding") or {},
            }
            case_id = f"{conversation_id}::T{turn_index:04d}"
            cases.append(
                {
                    "case_id": case_id,
                    "conversation_id": conversation_id,
                    "turn_index": turn_index,
                    "evidence": customer_judge_evidence(
                        str(turn.get("message") or ""),
                        result,
                        history=history,
                        source_reference=turn.get("source_context_tail") or [],
                    ),
                }
            )
            history.extend(
                [
                    {"role": "user", "content": str(turn.get("message") or "")},
                    {"role": "assistant", "content": str(turn.get("reply") or "")[:1000]},
                ]
            )
    return cases


def _read_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    verdicts: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return verdicts
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("judge_mode") != "llm":
                continue
            for item in record.get("items") or []:
                verdicts[str(item["case_id"])] = item
    return verdicts


def judge_batch(cases: list[dict[str, Any]]) -> dict[str, Any]:
    expected_ids = {str(item["case_id"]) for item in cases}
    try:
        raw = invoke_judge_stream(
            customer_batch_judge_prompt(cases),
            system=_CUSTOMER_JUDGE_SYSTEM,
            thinking=False,
            max_tokens=max(3000, len(cases) * 140),
        )
        items = validate_customer_batch_judge_output(raw, expected_ids)
        return {"judge_mode": "llm", "items": items}
    except Exception as exc:
        return {
            "judge_mode": "fallback",
            "items": [],
            "error": f"{type(exc).__name__}: {str(exc)[:600]}",
            "case_ids": sorted(expected_ids),
        }


def apply_verdicts(
    records: list[dict[str, Any]], verdicts: dict[str, dict[str, Any]]
) -> None:
    for conversation in records:
        conversation_id = str(conversation["conversation_id"])
        for turn in conversation.get("turns") or []:
            case_id = f"{conversation_id}::T{int(turn.get('turn_index') or 0):04d}"
            verdict = verdicts.get(case_id)
            if verdict is None:
                continue
            semantic = {key: value for key, value in verdict.items() if key != "case_id"}
            semantic.update(
                {
                    "judge_mode": "llm",
                    "judge_attempts": 1,
                    "output_repaired": False,
                    "batch_judge": True,
                }
            )
            turn["semantic_judge"] = semantic
            turn.setdefault("evaluation_evidence", {})["llm_judge"] = llm_judge_evidence(
                semantic, selected=True, requested=True
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch real LLM Judge over checkpointed Agent replay turns"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--min-batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--apply-checkpoint-only",
        action="store_true",
        help=(
            "Apply existing real-LLM checkpoint verdicts and rebuild reports "
            "without making any model request"
        ),
    )
    args = parser.parse_args()

    results_path = args.run_dir / "results.jsonl"
    manifest_path = args.run_dir / "manifest.json"
    checkpoint = args.run_dir / "judge_batch_results.jsonl"
    records = _read_records(results_path)
    cases = build_judge_cases(records)
    verdicts = _read_checkpoint(checkpoint if args.resume else Path("__missing__"))
    pending = [item for item in cases if str(item["case_id"]) not in verdicts]
    batch_size = max(1, args.batch_size)
    batches = [pending[index:index + batch_size] for index in range(0, len(pending), batch_size)]
    mode = "a" if args.resume and checkpoint.exists() else "w"
    minimum = max(1, min(args.min_batch_size, batch_size))

    if args.apply_checkpoint_only:
        apply_verdicts(records, verdicts)
        _atomic_write(results_path, records)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["batch_judge"] = {
            "checkpoint_applied_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "checkpoint_only": True,
            "total_turns": len(cases),
            "llm_verdicts": len(verdicts),
            "unresolved": len(pending),
            "configuration": judge_config_status(),
            "agent_rerun": False,
        }
        summary = write_replay_reports(args.run_dir, manifest, records)
        print(
            json.dumps(
                {
                    "mode": "apply_checkpoint_only",
                    "total_turns": len(cases),
                    "llm_verdicts": len(verdicts),
                    "unresolved": len(pending),
                    "summary": summary,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    with checkpoint.open(mode, encoding="utf-8") as stream:
        pending_batches = batches
        while pending_batches:
            wave = pending_batches
            pending_batches = []
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                futures = {executor.submit(judge_batch, batch): batch for batch in wave}
                for future in as_completed(futures):
                    batch = futures[future]
                    result = future.result()
                    if result.get("judge_mode") == "llm":
                        record = {
                            "batch_id": _batch_id(batch),
                            "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                            "batch_size": len(batch),
                            **result,
                        }
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        for item in result["items"]:
                            verdicts[str(item["case_id"])] = item
                        continue
                    if len(batch) > minimum:
                        midpoint = max(minimum, len(batch) // 2)
                        pending_batches.extend(
                            part for part in (batch[:midpoint], batch[midpoint:]) if part
                        )
                        continue
                    stream.write(
                        json.dumps(
                            {
                                "batch_id": _batch_id(batch),
                                "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                                "batch_size": len(batch),
                                **result,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    stream.flush()

    unresolved = len(cases) - len(verdicts)
    apply_verdicts(records, verdicts)
    _atomic_write(results_path, records)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["batch_judge"] = {
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "batch_size": batch_size,
        "workers": max(1, args.workers),
        "total_turns": len(cases),
        "llm_verdicts": len(verdicts),
        "unresolved": unresolved,
        "configuration": judge_config_status(),
        "agent_rerun": False,
    }
    summary = write_replay_reports(args.run_dir, manifest, records)
    print(
        json.dumps(
            {
                "total_turns": len(cases),
                "llm_verdicts": len(verdicts),
                "unresolved": unresolved,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.require_complete and unresolved:
        raise SystemExit(f"Batch Judge incomplete: unresolved={unresolved}")


if __name__ == "__main__":
    main()
