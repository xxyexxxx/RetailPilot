from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from RetailPilot.conversation_replay import apply_project_scope, llm_judge_evidence
from RetailPilot.customer_eval import judge_customer_turn
from scripts.run_multiturn_replay import write_replay_reports


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_write(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def retry_fallback_turns(
    records: list[dict[str, Any]],
    *,
    max_passes: int,
    delay_seconds: float,
) -> dict[str, int]:
    apply_project_scope(records)
    attempted = 0
    recovered = 0
    for _pass_index in range(max(1, max_passes)):
        changed = 0
        for conversation in records:
            history: list[dict[str, str]] = []
            for turn in conversation.get("turns") or []:
                semantic = turn.get("semantic_judge") or {}
                if (turn.get("scope_gate") or {}).get("in_scope") is False:
                    turn["semantic_judge"] = {
                        "judge_mode": "scope_filtered",
                        "pass": None,
                        "reason": (turn.get("scope_gate") or {}).get("reason"),
                    }
                    turn.setdefault("evaluation_evidence", {})["llm_judge"] = (
                        llm_judge_evidence(
                            turn["semantic_judge"], selected=False, requested=True
                        )
                    )
                elif semantic.get("judge_mode") == "deterministic_fallback":
                    attempted += 1
                    verdict = judge_customer_turn(
                        str(turn.get("message") or ""),
                        {
                            "reply": str(turn.get("reply") or ""),
                            "intents": turn.get("intents") or [],
                            "task_results": [],
                        },
                        history=history,
                        use_llm=True,
                    )
                    turn["semantic_judge"] = verdict
                    turn.setdefault("evaluation_evidence", {})["llm_judge"] = (
                        llm_judge_evidence(verdict, selected=True, requested=True)
                    )
                    if verdict.get("judge_mode") == "llm":
                        recovered += 1
                        changed += 1
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)
                history.extend(
                    [
                        {"role": "user", "content": str(turn.get("message") or "")},
                        {
                            "role": "assistant",
                            "content": str(turn.get("reply") or "")[:1000],
                        },
                    ]
                )
        if not changed:
            break
    unresolved = sum(
        1
        for conversation in records
        for turn in conversation.get("turns") or []
        if (turn.get("semantic_judge") or {}).get("judge_mode")
        == "deterministic_fallback"
    )
    return {"attempted": attempted, "recovered": recovered, "unresolved": unresolved}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retry only fallback LLM Judge turns without rerunning the Agent"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    results_path = args.run_dir / "results.jsonl"
    manifest_path = args.run_dir / "manifest.json"
    records = _read_records(results_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = retry_fallback_turns(
        records,
        max_passes=args.max_passes,
        delay_seconds=max(0.0, args.delay_seconds),
    )
    _atomic_write(results_path, records)
    manifest["judge_retry"] = {
        **stats,
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "agent_rerun": False,
    }
    summary = write_replay_reports(args.run_dir, manifest, records)
    print(json.dumps({**stats, "summary": summary}, ensure_ascii=False, indent=2))
    if args.require_complete and stats["unresolved"]:
        raise SystemExit(
            f"Judge retry incomplete: unresolved={stats['unresolved']}"
        )


if __name__ == "__main__":
    main()
