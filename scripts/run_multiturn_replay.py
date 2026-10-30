from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from RetailPilot.agent import chat
from RetailPilot.config import settings
from RetailPilot.conversation_replay import (
    REPLAY_VERSION,
    SCOPE_GATE_VERSION,
    apply_project_scope,
    badcase_turns,
    cluster_badcases,
    filter_conversations_by_project_scope,
    grade_replay_turn_detailed,
    judge_selection,
    llm_judge_evidence,
    load_conversations,
    project_scope_gate,
    replay_conversation_is_badcase,
    summarize_replay,
    scope_filtered_turns,
)
from RetailPilot.customer_eval import judge_customer_turn
from RetailPilot.llm_config import agent_model_status, judge_config_status


def _run_conversation(
    conversation: dict[str, Any],
    *,
    run_id: str,
    max_user_turns: int | None,
    use_llm_judge: bool,
    judge_turns: set[tuple[str, int]],
) -> dict[str, Any]:
    conversation_id = str(conversation["conversation_id"])
    session_id = f"{run_id}-{conversation_id}"[-100:]
    stable_user = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:8]
    user_id = f"replay-{stable_user}"
    records: list[dict[str, Any]] = []
    source_context: list[dict[str, str]] = []
    agent_history: list[dict[str, str]] = []
    previous_reply = ""
    previous_message = ""
    user_turn_index = 0
    for source_turn in conversation.get("turns") or []:
        if source_turn["role"] != "user":
            source_context.append(
                {"role": "assistant_reference", "content": source_turn["content"]}
            )
            continue
        user_turn_index += 1
        if max_user_turns is not None and user_turn_index > max_user_turns:
            break
        message = str(source_turn["content"])
        scope_gate = source_turn.get("scope_gate") or project_scope_gate(message)
        started = time.perf_counter()
        try:
            result = chat(session_id, message, user_id)
            hard_evidence = grade_replay_turn_detailed(
                result,
                previous_reply=previous_reply,
                previous_message=previous_message,
                current_message=message,
            )
            selected = bool(scope_gate["in_scope"]) and (
                conversation_id, user_turn_index
            ) in judge_turns
            semantic = (
                judge_customer_turn(
                    message,
                    result,
                    history=agent_history,
                    source_reference=source_context[-4:],
                    use_llm=use_llm_judge,
                )
                if selected
                else (
                    {"judge_mode": "not_selected", "pass": None}
                    if scope_gate["in_scope"]
                    else {
                        "judge_mode": "scope_filtered",
                        "pass": None,
                        "reason": scope_gate["reason"],
                    }
                )
            )
            judge_evidence = llm_judge_evidence(
                semantic,
                selected=selected,
                requested=use_llm_judge,
            )
            reply = str(result.get("reply") or "")
            tools = [
                item.get("tool")
                for item in result.get("trace") or []
                if item.get("node") == "tool_execution"
            ]
            record = {
                "turn_index": user_turn_index,
                "source_turn_index": source_turn.get("source_turn_index", user_turn_index),
                "message": message,
                "reply": reply,
                "hard_pass": hard_evidence["pass"],
                "hard_failure_types": hard_evidence["failure_types"],
                "scope_gate": scope_gate,
                "semantic_judge": semantic,
                "evaluation_evidence": {
                    "scope_gate": scope_gate,
                    "hard_check": hard_evidence,
                    "llm_judge": judge_evidence,
                },
                "intents": result.get("intents") or [],
                "tools": tools,
                "needs_clarification": bool(result.get("needs_clarification")),
                "grounding": result.get("grounding") or {},
                "task_results": result.get("task_results") or [],
                "trace": result.get("trace") or [],
                "source_action": source_turn.get("source_action") or "",
                "source_satisfaction": source_turn.get("source_satisfaction") or [],
                "source_context_tail": source_context[-4:],
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            previous_reply = reply
            agent_history.extend(
                [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": reply[:1000]},
                ]
            )
        except Exception as exc:
            selected = bool(scope_gate["in_scope"]) and (
                conversation_id, user_turn_index
            ) in judge_turns
            semantic = (
                {
                    "judge_mode": "not_run",
                    "pass": False,
                    "failure_type": "system_error",
                }
                if scope_gate["in_scope"]
                else {
                    "judge_mode": "scope_filtered",
                    "pass": None,
                    "reason": scope_gate["reason"],
                }
            )
            hard_evidence = {
                "checker": "program_hard_check",
                "pass": False,
                "failure_types": ["system_error"],
                "checks": [
                    {
                        "name": "agent_execution",
                        "pass": False,
                        "evidence": {
                            "error": f"{type(exc).__name__}: {str(exc)[:500]}"
                        },
                    }
                ],
            }
            record = {
                "turn_index": user_turn_index,
                "source_turn_index": source_turn.get("source_turn_index", user_turn_index),
                "message": message,
                "reply": "",
                "hard_pass": False,
                "hard_failure_types": ["system_error"],
                "scope_gate": scope_gate,
                "semantic_judge": semantic,
                "evaluation_evidence": {
                    "scope_gate": scope_gate,
                    "hard_check": hard_evidence,
                    "llm_judge": llm_judge_evidence(
                        semantic,
                        selected=selected,
                        requested=use_llm_judge,
                    ),
                },
                "intents": [],
                "tools": [],
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                "source_action": source_turn.get("source_action") or "",
                "source_satisfaction": source_turn.get("source_satisfaction") or [],
                "source_context_tail": source_context[-4:],
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        records.append(record)
        previous_message = message
        source_context.append({"role": "user", "content": message})
    return {
        "conversation_id": conversation_id,
        "source": conversation.get("source") or "",
        "metadata": conversation.get("metadata") or {},
        "user_turns_replayed": len(records),
        "turns": records,
    }


def _badcase(record: dict[str, Any]) -> bool:
    return replay_conversation_is_badcase(record)


def _rate(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.1%}"


def _markdown(manifest: dict[str, Any], summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Multi-turn Conversation Shadow Replay",
            "",
            f"- Run: `{manifest['run_id']}`",
            f"- Source / format: `{manifest['source']}` / `{manifest['input_format']}`",
            f"- Conversations / user turns: {summary['conversations']} / {summary['turns']}",
            f"- In-scope / scope-filtered turns: {summary['in_scope_turns']} / "
            f"{summary['scope_filtered_turns']}",
            f"- Scope filter reasons: `{json.dumps(summary['scope_filter_counts'], ensure_ascii=False)}`",
            f"- Hard pass (in-scope only): {summary['hard_turns_passed']}/{summary['in_scope_turns']} "
            f"({summary['hard_pass_rate']:.1%})",
            f"- Real LLM Judge calls / evaluable / not-evaluable: "
            f"{summary['llm_judged_turns']} / {summary['llm_evaluable_turns']} / "
            f"{summary['llm_not_evaluable_turns']}；pass "
            f"{summary['llm_turns_passed']} ({_rate(summary['llm_pass_rate'])})",
            f"- LLM Judge effective: `{str(summary['llm_judge_effective']).lower()}`",
            f"- Deterministic fallback / offline sidecar: "
            f"{summary['fallback_turns']} / {summary['deterministic_sidecar_turns']} "
            "(excluded from LLM metrics)",
            f"- Judge modes: `{json.dumps(summary['judge_mode_counts'], ensure_ascii=False)}`",
            f"- Badcase conversations: {summary['badcase_conversations']}",
            f"- Hard failures: `{json.dumps(summary['hard_failure_counts'], ensure_ascii=False)}`",
            f"- Real LLM failures: `{json.dumps(summary['semantic_failure_counts'], ensure_ascii=False)}`",
            f"- Intent counts: `{json.dumps(summary['intent_counts'], ensure_ascii=False)}`",
            f"- Tool counts: `{json.dumps(summary['tool_call_counts'], ensure_ascii=False)}`",
            f"- P50 / P95: {summary['latency_p50_ms']} / {summary['latency_p95_ms']} ms",
            "",
            "> 只有 `judge_mode=llm` 计入 Real LLM Judge。`deterministic_fallback` 和",
            "> `deterministic` 仅为程序侧诊断，不计作 LLM 评估，也不影响 LLM 通过率。",
            "> `scope_filtered` 表示不属于当前项目范围，不送 LLM Judge，",
            "> 不算通过或失败，也不进入 Badcase 修复队列。",
            "",
            "> 这是用户 Query 分布的 shadow replay，不是线上准确率。原始人工客服回复只保存在",
            "> source_context_tail 供归因参考，不注入 Agent，也不作为商品、订单或政策事实 Gold。",
            "",
        ]
    )


_ENGINEERING_LAYER = {
    "duplicate_reply": "Context / Response policy",
    "duplicate_task": "Planner / Task deduplication",
    "empty_reply": "Runtime / Response contract",
    "system_error": "Runtime reliability",
    "grounding_failure": "Grounding / Citation",
    "unsupported_claim": "Grounding / Citation",
    "citation_mismatch": "Grounding / Citation",
    "off_topic": "Intent routing / Response",
    "context_miss": "Context engineering",
    "incomplete": "Response synthesis",
}


def _analysis_markdown(
    summary: dict[str, Any], clusters: list[dict[str, Any]]
) -> str:
    lines = [
        "# Badcase Analysis",
        "",
        f"- In-scope / scope-filtered turns: {summary['in_scope_turns']} / {summary['scope_filtered_turns']}",
        f"- In-scope hard-check failed turns: {summary['in_scope_turns'] - summary['hard_turns_passed']}",
        f"- Real LLM calls / evaluable / failed: {summary['llm_judged_turns']} / "
        f"{summary['llm_evaluable_turns']} / "
        f"{summary['llm_evaluable_turns'] - summary['llm_turns_passed']}",
        f"- Shadow turns marked not-evaluable: {summary['llm_not_evaluable_turns']}",
        f"- Fallback turns excluded from LLM results: {summary['fallback_turns']}",
        "",
        "| Evaluator | Failure | Harness signature | Engineering layer | Turns | Conversations |",
        "|---|---|---|---|---:|---:|",
    ]
    lines.extend(
        f"| {item['evaluator']} | {item['failure_type']} | `{item.get('routing_signature', '-')}` | "
        f"{_ENGINEERING_LAYER.get(item['failure_type'], 'Needs triage')} | "
        f"{item['turn_count']} | {item['conversation_count']} |"
        for item in clusters
    )
    for cluster in clusters:
        failure = cluster["failure_type"]
        lines.extend(
            [
                "",
                f"## {cluster['evaluator']} / {failure}",
                "",
                f"- Suggested layer: {_ENGINEERING_LAYER.get(failure, 'Needs triage')}",
                f"- Harness signature: `{cluster.get('routing_signature', '-')}`",
                f"- Affected turns / conversations: {cluster['turn_count']} / "
                f"{cluster['conversation_count']}",
                "",
            ]
        )
        for sample in cluster["examples"]:
            judge_reason = sample.get("llm_judge_summary")
            reason = f"；LLM evidence: {judge_reason}" if judge_reason else ""
            lines.append(
                f"- `{sample['conversation_id']}/T{sample['turn_index']}` "
                f"{sample['message']} → {str(sample.get('reply') or '')[:160]}{reason}"
            )
    if not clusters:
        lines.extend(["", "- 本轮未发现硬异常或 Judge Badcase。"])
    return "\n".join(lines) + "\n"


def _read_existing(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                records[str(record["conversation_id"])] = record
    return records


def _canonical_conversation_id(value: object) -> str:
    """Collapse replay-generated duplicate scope suffixes.

    Replaying an already scope-segmented corpus can append the same
    ``::scope-NNN`` suffix again. Badcase checkpoints must still map back to
    the source segment without requiring users to rewrite artifact IDs.
    """

    parts = str(value or "").strip().split("::")
    canonical: list[str] = []
    for part in parts:
        if canonical and part.startswith("scope-") and part == canonical[-1]:
            continue
        canonical.append(part)
    return "::".join(canonical)


def load_badcase_conversation_ids(path: Path) -> set[str]:
    """Load the conversation set for a zero-cost, cluster-focused replay."""

    conversation_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid badcase JSONL at {path}:{line_number}: {exc}"
                ) from exc
            conversation_id = _canonical_conversation_id(record.get("conversation_id"))
            if conversation_id:
                conversation_ids.add(conversation_id)
    if not conversation_ids:
        raise ValueError(f"badcase file contains no conversation_id: {path}")
    return conversation_ids


def write_replay_reports(
    output: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Regenerate every report from checkpointed conversation records."""

    apply_project_scope(records)
    summary = summarize_replay(records)
    source_scope = manifest.get("pre_agent_scope_filter") or {}
    if source_scope:
        summary.update(
            {
                "source_user_turns": source_scope.get("source_user_turns", 0),
                "pre_agent_scope_eligible_turns": source_scope.get(
                    "eligible_user_turns", 0
                ),
                "pre_agent_scope_filtered_turns": source_scope.get(
                    "scope_filtered_turns", 0
                ),
                "pre_agent_scope_filter_counts": source_scope.get(
                    "scope_filter_counts", {}
                ),
            }
        )
    badcases = [record for record in records if _badcase(record)]
    failed_turns = badcase_turns(records)
    badcase_clusters = cluster_badcases(records)
    filtered_turns = scope_filtered_turns(records)
    manifest.update(
        {
            "llm_judge_effective": summary["llm_judge_effective"],
            "llm_judged_turns": summary["llm_judged_turns"],
            "fallback_turns_excluded_from_llm_metrics": summary["fallback_turns"],
            "scope_gate_version": SCOPE_GATE_VERSION,
            "scope_filtered_turns": summary["scope_filtered_turns"],
        }
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "badcases.jsonl").open("w", encoding="utf-8") as stream:
        for record in badcases:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output / "badcase_turns.jsonl").open("w", encoding="utf-8") as stream:
        for record in failed_turns:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output / "scope_filtered_turns.jsonl").open("w", encoding="utf-8") as stream:
        for record in filtered_turns:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output / "badcase_clusters.json").write_text(
        json.dumps(badcase_clusters, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "summary.md").write_text(_markdown(manifest, summary), encoding="utf-8")
    (output / "badcase_analysis.md").write_text(
        _analysis_markdown(summary, badcase_clusters), encoding="utf-8"
    )
    return summary


def write_scope_filter_outputs(
    output: Path,
    eligible_conversations: list[dict[str, Any]],
    filtered_turns: list[dict[str, Any]],
    scope_summary: dict[str, Any],
) -> None:
    (output / "scope_summary.json").write_text(
        json.dumps(scope_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "eligible_conversations.jsonl").open("w", encoding="utf-8") as stream:
        for conversation in eligible_conversations:
            stream.write(json.dumps(conversation, ensure_ascii=False) + "\n")
    with (output / "scope_filtered_turns.jsonl").open("w", encoding="utf-8") as stream:
        for turn in filtered_turns:
            stream.write(json.dumps(turn, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay real/custom multi-turn user queries and emit trace-aware badcases"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--format", dest="input_format", choices=["jddc-uss", "jsonl"], required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--min-user-turns", type=int, default=2)
    parser.add_argument("--max-user-turns", type=int, default=12)
    parser.add_argument(
        "--scope-only",
        action="store_true",
        help="Only build the project-scope corpus; do not call the Agent or Judge",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--llm-judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the independent LLM judge (default: on; --no-llm-judge disables)",
    )
    parser.add_argument(
        "--require-llm-judge",
        action="store_true",
        help="Exit non-zero unless every selected semantic verdict used a real LLM",
    )
    parser.add_argument(
        "--judge-limit", type=int, default=0,
        help="Max turns selected for semantic judge; 0 means all replayed turns",
    )
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--badcase-file",
        type=Path,
        help="Replay only conversations referenced by an existing badcase JSONL file",
    )
    args = parser.parse_args()

    run_id = f"replay-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    output = args.output_dir or settings.artifacts_dir / "conversation_replay_runs" / run_id
    output.mkdir(parents=True, exist_ok=args.resume)
    results_path = output / "results.jsonl"
    conversations = load_conversations(
        args.input,
        input_format=args.input_format,
        limit=max(1, args.limit),
        min_user_turns=max(1, args.min_user_turns),
    )
    selected_badcase_ids: set[str] = set()
    if args.badcase_file:
        selected_badcase_ids = load_badcase_conversation_ids(args.badcase_file)
        conversations = [
            item
            for item in conversations
            if _canonical_conversation_id(item["conversation_id"]) in selected_badcase_ids
        ]
        if not conversations:
            raise SystemExit("--badcase-file did not match any input conversation")
    max_user_turns = None if args.max_user_turns <= 0 else args.max_user_turns
    conversations, prefiltered_turns, scope_summary = filter_conversations_by_project_scope(
        conversations,
        max_user_turns=max_user_turns,
    )
    write_scope_filter_outputs(output, conversations, prefiltered_turns, scope_summary)
    if args.scope_only:
        scope_manifest = {
            "run_id": run_id,
            "replay_version": REPLAY_VERSION,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "source": str(args.input.resolve()),
            "input_format": args.input_format,
            "scope_only": True,
            "agent_executed": False,
            "llm_judge_executed": False,
            "raw_data_committed": False,
            "scope_filter": scope_summary,
        }
        (output / "manifest.json").write_text(
            json.dumps(scope_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({"output": str(output), **scope_summary}, ensure_ascii=False, indent=2))
        return
    if args.llm_judge:
        judge_status = judge_config_status()
        if not judge_status.get("configured"):
            raise SystemExit(
                "LLM judge is mandatory by default and no independent JUDGE_API_KEY "
                f"is configured: {judge_status.get('error') or 'unknown error'}. "
                "Set JUDGE_* explicitly or pass --no-llm-judge to opt out."
            )
    judge_turns = judge_selection(
        conversations,
        judge_limit=max(0, args.judge_limit),
        seed=args.seed,
        max_user_turns=None,
    )
    existing = _read_existing(results_path) if args.resume else {}
    pending = [
        item for item in conversations if str(item["conversation_id"]) not in existing
    ]

    def run_one(item: dict[str, Any]) -> dict[str, Any]:
        return _run_conversation(
            item,
            run_id=run_id,
            max_user_turns=max_user_turns,
            use_llm_judge=args.llm_judge,
            judge_turns=judge_turns,
        )

    mode = "a" if args.resume and results_path.exists() else "w"
    with results_path.open(mode, encoding="utf-8") as stream:
        if args.workers <= 1:
            iterator = (run_one(item) for item in pending)
            for record in iterator:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                existing[record["conversation_id"]] = record
        else:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                futures = {executor.submit(run_one, item): item for item in pending}
                for future in as_completed(futures):
                    record = future.result()
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    existing[record["conversation_id"]] = record

    records = [existing[str(item["conversation_id"])] for item in conversations]
    manifest = {
        "run_id": run_id,
        "replay_version": REPLAY_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": str(args.input.resolve()),
        "input_format": args.input_format,
        "limit": args.limit,
        "max_user_turns": args.max_user_turns,
        "pre_agent_scope_filter": scope_summary,
        "workers": args.workers,
        "llm_judge_requested": args.llm_judge,
        "evaluation_contract": {
            "hard_check": "deterministic program assertions",
            "scope_gate": (
                "project-scope eligible turns only; filtered turns are neither pass nor badcase"
            ),
            "llm_judge": "only judge_mode=llm counts",
            "deterministic_fallback": "diagnostic only; excluded from LLM metrics",
        },
        "system_under_test": {
            "llm_mode": settings.llm_mode,
            "intent_mode": settings.intent_mode,
            "agent_model": agent_model_status(),
            "judge_isolated_from_agent_generation": (
                settings.llm_mode == "off" and settings.intent_mode == "rules"
            ),
        },
        "judge_configuration": (
            judge_config_status() if args.llm_judge else {"configured": False, "requested": False}
        ),
        "judge_limit": args.judge_limit,
        "shadow_replay": True,
        "raw_data_committed": False,
        "badcase_replay": {
            "source": str(args.badcase_file.resolve()) if args.badcase_file else None,
            "requested_conversations": len(selected_badcase_ids),
            "matched_conversations": len(conversations),
        },
    }
    summary = write_replay_reports(output, manifest, records)
    # ``write_replay_reports`` regenerates its compatibility filtered file from
    # replay records. Restore the true pre-Agent filter artifact afterwards.
    write_scope_filter_outputs(output, conversations, prefiltered_turns, scope_summary)
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))
    if args.require_llm_judge:
        if not args.llm_judge:
            raise SystemExit("--require-llm-judge also requires --llm-judge")
        selected_count = len(judge_turns)
        if summary["llm_judged_turns"] != selected_count:
            raise SystemExit(
                "Real LLM Judge requirement failed: "
                f"selected={selected_count}, real_llm={summary['llm_judged_turns']}, "
                f"judge modes={summary['judge_mode_counts']}"
            )


if __name__ == "__main__":
    main()
