"""Compare Replay artifacts, or run a clearly labelled offline control-flow demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from RetailPilot.evaluation import evaluate_policy_runs, policy_quality_gate
from RetailPilot.policy_followup import run_policy_followup


def mock_records(enabled: bool) -> list[dict]:
    query = "定制商品超过七天出现质量问题还能退货吗？"
    documents = [
        {
            "chunk_id": "general",
            "doc_id": "returns-v1",
            "topic": "退换货",
            "content": "定制商品不适用七天无理由退货。",
        },
        {
            "chunk_id": "exception",
            "doc_id": "quality-v1",
            "topic": "保修",
            "content": "超过七天出现质量问题可申请核验，是否退货须按检测结果审核。",
        },
    ]

    def assess(question, *, topic, documents, **kw):
        ids = {item["chunk_id"] for item in documents}
        complete = ids == {"general", "exception"}
        mapping = {"定制限制": ["general"]}
        if complete:
            mapping["质量例外与期限"] = ["exception"]
        return {
            "decision": "answer" if complete else "clarify",
            "next_action": "answer" if complete else "retrieve_more",
            "sufficient": complete,
            "conflict": False,
            "reason": "固定标注的离线流程夹具，不是模型语义判断",
            "covered_aspects": list(mapping),
            "missing_aspects": [] if complete else ["质量例外与期限"],
            "supporting_chunk_ids": sorted(ids),
            "aspect_evidence_map": mapping,
            "suggested_queries": []
            if complete
            else [
                {
                    "aspect": "质量例外与期限",
                    "topic": "保修",
                    "query": query + "；补充核对：质量例外与期限",
                }
            ],
        }

    result = run_policy_followup(
        query=query,
        topic="退换货",
        initial_payload={"documents": documents[:1]},
        retrieval_arguments={"platform": "demo-mall", "limit": 6},
        assess=assess,
        enabled=enabled,
        retrieve=lambda args: {
            "ok": True,
            "data": {"documents": documents[1:]},
            "trace": {"node": "tool_execution"},
        },
    )
    return [
        {
            "conversation_id": "mock-policy",
            "turns": [
                {
                    "message": query,
                    "trace": result["trace"],
                    "expected_policy": {
                        "required_chunk_ids": ["general", "exception"],
                        "expected_action": "answer",
                    },
                    "response": "\n".join(
                        item["content"] for item in result["documents"]
                    )
                    if result["documents"]
                    else "证据不足，暂不直接给出结论。",
                }
            ],
        }
    ]


def load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--enhanced", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "policy-adaptive"
    )
    args = parser.parse_args()
    if not args.mock and not (args.baseline and args.enhanced):
        parser.error("provide --mock or both Replay JSONL paths")
    baseline = mock_records(False) if args.mock else load(args.baseline)
    enhanced = mock_records(True) if args.mock else load(args.enhanced)
    report = {
        "mode": "mock_control_flow_only" if args.mock else "replay_artifacts",
        "baseline": evaluate_policy_runs(baseline),
        "enhanced": evaluate_policy_runs(enhanced),
        "limitations": "Mock verdicts do not validate semantic accuracy. Missing judge, cost or filter labels are null.",
    }
    gates = json.loads(
        (ROOT / "config" / "quality-gates.json").read_text(encoding="utf-8")
    )["policy_required"]
    report["quality_gate"] = policy_quality_gate(report["enhanced"], gates)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "example-trace.json").write_text(
        json.dumps(enhanced, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
