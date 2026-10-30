from __future__ import annotations

import json
import math
import sqlite3
import statistics
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .catalog import list_brands
from .config import settings
from .database import db_session, json_dumps, json_loads
from .query import parse_query
from .retrieval import search_engine
from .text import tokenize


def _dcg(relevances: list[float]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(relevances))


def _ndcg(
    relevances: list[float], ideal_relevances: list[float] | None = None
) -> float:
    if not relevances:
        return 0.0
    ideal = sorted(ideal_relevances or relevances, reverse=True)[:10]
    denominator = _dcg(ideal)
    return _dcg(relevances) / denominator if denominator else 0.0


def _attribute_case_failure(
    parsed: dict[str, Any],
    expected: dict[str, Any],
    products: list[dict[str, Any]],
    judge: dict[str, Any],
    candidate_count: int = 0,
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    if expected.get("intent") and parsed["intent"] != expected["intent"]:
        failures.append(
            {
                "stage": "intent",
                "reason": f"expected {expected['intent']}, got {parsed['intent']}",
            }
        )
    if expected.get("category") and parsed.get("category") != expected["category"]:
        failures.append(
            {
                "stage": "query_understand",
                "reason": f"expected category {expected['category']}, got {parsed.get('category')}",
            }
        )
    if expected.get("sort") and parsed.get("sort") != expected["sort"]:
        failures.append(
            {
                "stage": "query_understand",
                "reason": f"expected sort {expected['sort']}, got {parsed.get('sort')}",
            }
        )
    if expected.get("clarification_slot") != parsed.get("clarification_slot"):
        if expected.get("clarification_slot"):
            failures.append(
                {
                    "stage": "clarify",
                    "reason": (
                        f"expected clarification {expected['clarification_slot']}, "
                        f"got {parsed.get('clarification_slot')}"
                    ),
                }
            )
    price_max = expected.get("price_max")
    if price_max is not None and any(product["price"] > price_max for product in products):
        failures.append({"stage": "structured_filter", "reason": "price constraint violated"})
    negative_brand = expected.get("negative_brand")
    if negative_brand and str(negative_brand).lower() not in {
        str(brand).lower() for brand in parsed.get("negative_brands") or []
    }:
        failures.append(
            {
                "stage": "query_understand",
                "reason": f"negative brand {negative_brand} was not extracted",
            }
        )
    if negative_brand and any(
        product["brand"].lower() == str(negative_brand).lower() for product in products
    ):
        failures.append({"stage": "structured_filter", "reason": "negative brand returned"})
    preferred_brand = expected.get("preferred_brand")
    if preferred_brand and str(preferred_brand).lower() not in {
        str(brand).lower() for brand in parsed.get("preferred_brands") or []
    }:
        failures.append(
            {
                "stage": "query_understand",
                "reason": f"preferred brand {preferred_brand} was not extracted",
            }
        )
    parsed_negative_terms = {
        str(term).lower() for term in parsed.get("negative_terms") or []
    }
    for term in expected.get("negative_title_terms") or []:
        if str(term).lower() not in parsed_negative_terms:
            failures.append(
                {
                    "stage": "query_understand",
                    "reason": f"negative term {term} was not extracted",
                }
            )
        if any(str(term).lower() in product["title"].lower() for product in products):
            failures.append(
                {
                    "stage": "structured_filter",
                    "reason": f"negative title term {term} returned",
                }
            )
    if len(products) < int(expected.get("min_results", 0)):
        failures.append(
            {
                "stage": "recall",
                "reason": f"expected at least {expected['min_results']} results, got {len(products)}",
            }
        )
    min_candidate_count = expected.get("min_candidate_count")
    if min_candidate_count is not None and candidate_count < int(min_candidate_count):
        failures.append(
            {
                "stage": "recall",
                "reason": (
                    f"expected at least {min_candidate_count} candidates, "
                    f"got {candidate_count}"
                ),
            }
        )
    forbidden_title_terms = [
        str(term).lower() for term in (expected.get("forbidden_title_terms") or [])
    ]
    for term in forbidden_title_terms:
        if any(term in product["title"].lower() for product in products):
            failures.append(
                {
                    "stage": "structured_filter",
                    "reason": f"forbidden title term {term} returned",
                }
            )
    if not judge.get("pass", True):
        failures.append(
            {
                "stage": "fact_grounding",
                "reason": "; ".join(judge.get("issues") or ["citation check failed"]),
            }
        )
    return failures


def load_cases() -> list[dict[str, Any]]:
    with db_session() as connection:
        rows = connection.execute(
            """
            SELECT case_id, name, query, context_json, expected_json, tags_json
            FROM eval_cases
            WHERE enabled = 1
            ORDER BY case_id
            """
        ).fetchall()
    return [
        {
            "case_id": row["case_id"],
            "name": row["name"],
            "query": row["query"],
            "context": json_loads(row["context_json"], {}),
            "expected": json_loads(row["expected_json"], {}),
            "tags": json_loads(row["tags_json"], []),
        }
        for row in rows
    ]


def load_qrels() -> dict[str, dict[str, int]]:
    """Load graded relevance from the physically separated holdout artifact.

    The main product database stays the system under test; qrels are no longer
    its ``eval_qrels`` table when the holdout database exists.  The fallback
    below only preserves compatibility with old local checkouts.
    """

    rows: list[tuple[str, str, int]] = []
    if settings.eval_holdout_path.exists():
        with sqlite3.connect(settings.eval_holdout_path) as connection:
            rows = connection.execute(
                "SELECT case_id, sku, relevance FROM eval_qrels"
            ).fetchall()
    if not rows:
        with db_session() as connection:
            rows = connection.execute(
                "SELECT case_id, sku, relevance FROM eval_qrels"
            ).fetchall()
    qrels: dict[str, dict[str, int]] = {}
    for row in rows:
        case_id, sku, relevance = row[0], row[1], int(row[2])
        qrels.setdefault(str(case_id), {})[str(sku)] = relevance
    return qrels


def evaluate_mode(
    mode: str,
    cases: list[dict[str, Any]],
    qrels: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    qrels = qrels or load_qrels()
    brands = list_brands()
    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    for case in cases:
        parsed = parse_query(case["query"], case["context"], brands)
        expected = case["expected"]
        executed_search = (
            parsed.intent in {"search", "compare", "detail"}
            and not parsed.clarification_slot
        )
        search_result: dict[str, Any] = {
            "products": [],
            "judge": {
                "pass": False,
                "issues": [{"type": "search_not_run", "reason": "case was not executed"}],
            },
            "latency_ms": 0.0,
        }
        if executed_search:
            search_result = search_engine.search(
                parsed,
                user_id="new-user",
                limit=10,
                mode=mode,
                session_id=f"eval-{mode}",
            )
            latencies.append(float(search_result["latency_ms"]))
        products = search_result["products"]
        case_qrels = qrels.get(case["case_id"], {})
        if case_qrels:
            relevances = [float(case_qrels.get(product["sku"], 0)) for product in products]
            ideal_relevances = sorted(case_qrels.values(), reverse=True)[:10]
        else:
            # Rule-based product relevance must not act as a gold standard.
            relevances = [0.0 for _ in products]
            ideal_relevances = []
        retrieval = search_result.get("retrieval") or {}
        raw_candidates = retrieval.get("raw_candidate_ranking") or []
        relevant_skus = {sku for sku, relevance in case_qrels.items() if relevance >= 2}

        def candidate_recall(size: int) -> float:
            if not relevant_skus:
                return 0.0
            return len(set(raw_candidates[:size]) & relevant_skus) / len(relevant_skus)

        route_recalls = {}
        for route, ranking in (retrieval.get("routes") or {}).items():
            route_recalls[route] = (
                len(set(ranking[:100]) & relevant_skus) / len(relevant_skus)
                if relevant_skus
                else 0.0
            )
        failures: list[dict[str, str]] = []
        if not executed_search:
            failures.append(
                {
                    "stage": "not_evaluable",
                    "reason": "search engine did not execute this case",
                }
            )
        else:
            failures = _attribute_case_failure(
                parsed.to_dict(),
                expected,
                products,
                search_result["judge"],
                int(search_result.get("total_candidates") or 0),
            )
            if not case_qrels:
                failures.append(
                    {
                        "stage": "qrels_missing",
                        "reason": "no graded qrels for this case; rule self-grading is not counted",
                    }
                )
        recall_at_50 = candidate_recall(50)
        recall_at_100 = candidate_recall(100)
        if case_qrels and relevant_skus and recall_at_50 == 0:
            failures.append(
                {"stage": "recall", "reason": "no graded-relevant SKU recalled at candidate@50"}
            )
        expected_category = expected.get("category")
        precision = (
            sum(1 for product in products if product["category"] == expected_category)
            / len(products)
            if products and expected_category
            else 0.0
        )
        results.append(
            {
                **case,
                "parsed": parsed.to_dict(),
                "result_skus": [product["sku"] for product in products],
                "top_titles": [product["title"] for product in products[:3]],
                "latency_ms": search_result["latency_ms"],
                "ndcg_at_10": round(_ndcg(relevances, ideal_relevances), 4),
                "candidate_recall_at_50": round(recall_at_50, 4),
                "candidate_recall_at_100": round(recall_at_100, 4),
                "route_recall_at_100": {
                    route: round(value, 4) for route, value in route_recalls.items()
                },
                "category_precision_at_10": round(precision, 4),
                "fact_pass": bool(search_result["judge"].get("pass", False)),
                "pass": not failures,
                "failures": failures,
            }
        )

    def ratio(predicate) -> float:
        return (
            sum(1 for item in results if predicate(item)) / len(results)
            if results
            else 0.0
        )

    category_cases = [item for item in results if item["expected"].get("category")]
    search_cases = [
        item
        for item in results
        if item["parsed"]["intent"] in {"search", "compare", "detail"}
        and not item["parsed"].get("clarification_slot")
    ]
    qrel_cases = [item for item in search_cases if qrels.get(item["case_id"])]
    metrics = {
        "cases": len(results),
        "pass_rate": round(ratio(lambda item: item["pass"]), 4),
        "intent_accuracy": round(
            ratio(lambda item: item["parsed"]["intent"] == item["expected"].get("intent")), 4
        ),
        "category_accuracy": round(
            sum(
                1
                for item in category_cases
                if item["parsed"].get("category") == item["expected"].get("category")
            )
            / len(category_cases)
            if category_cases
            else 1.0,
            4,
        ),
        "constraint_pass_rate": round(
            ratio(
                lambda item: not any(
                    failure["stage"] == "structured_filter"
                    for failure in item["failures"]
                )
            ),
            4,
        ),
        "fact_pass_rate": round(ratio(lambda item: item["fact_pass"]), 4),
        "mean_ndcg_at_10": round(
            statistics.mean(item["ndcg_at_10"] for item in search_cases)
            if search_cases
            else 0.0,
            4,
        ),
        "mean_candidate_recall_at_50": round(
            statistics.mean(item["candidate_recall_at_50"] for item in qrel_cases)
            if qrel_cases
            else 0.0,
            4,
        ),
        "mean_candidate_recall_at_100": round(
            statistics.mean(item["candidate_recall_at_100"] for item in qrel_cases)
            if qrel_cases
            else 0.0,
            4,
        ),
        "route_recall_at_100": {
            route: round(
                statistics.mean(
                    item["route_recall_at_100"].get(route, 0.0) for item in qrel_cases
                ),
                4,
            )
            for route in ("fts", "dense", "popular", "personal")
        },
        "mean_category_precision_at_10": round(
            statistics.mean(item["category_precision_at_10"] for item in search_cases)
            if search_cases
            else 0.0,
            4,
        ),
        "zero_result_rate": round(
            sum(1 for item in search_cases if not item["result_skus"]) / len(search_cases)
            if search_cases
            else 0.0,
            4,
        ),
        "latency_p50_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "latency_p95_ms": round(
            sorted(latencies)[min(len(latencies) - 1, math.ceil(len(latencies) * 0.95) - 1)],
            2,
        )
        if latencies
        else 0.0,
    }
    root_causes: dict[str, int] = {}
    for item in results:
        for failure in item["failures"]:
            root_causes[failure["stage"]] = root_causes.get(failure["stage"], 0) + 1
    return {
        "mode": mode,
        "metrics": metrics,
        "root_causes": dict(sorted(root_causes.items(), key=lambda item: -item[1])),
        "cases": results,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 搜推评测报告",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- 生成时间: `{report['created_at']}`",
        f"- 用例数: `{report['case_count']}`",
        "",
        "## 指标对比",
        "",
        "| 指标 | FTS baseline | Hybrid |",
        "|---|---:|---:|",
    ]
    baseline = report["modes"]["fts"]["metrics"]
    hybrid = report["modes"]["hybrid"]["metrics"]
    metric_labels = {
        "pass_rate": "用例通过率",
        "intent_accuracy": "意图准确率",
        "category_accuracy": "类目归一准确率",
        "constraint_pass_rate": "硬约束通过率",
        "fact_pass_rate": "事实校验通过率",
        "mean_ndcg_at_10": "NDCG@10",
        "mean_candidate_recall_at_50": "候选 Recall@50",
        "mean_candidate_recall_at_100": "候选 Recall@100",
        "mean_category_precision_at_10": "类目 Precision@10",
        "zero_result_rate": "空召回率",
        "latency_p50_ms": "P50 延迟(ms)",
        "latency_p95_ms": "P95 延迟(ms)",
    }
    for key, label in metric_labels.items():
        lines.append(f"| {label} | {baseline[key]} | {hybrid[key]} |")
    for route in ("fts", "dense", "popular", "personal"):
        lines.append(
            f"| {route} 路 Recall@100 | "
            f"{baseline['route_recall_at_100'][route]} | "
            f"{hybrid['route_recall_at_100'][route]} |"
        )
    lines.extend(["", "## Hybrid Badcases", ""])
    failures = [item for item in report["modes"]["hybrid"]["cases"] if not item["pass"]]
    if not failures:
        lines.append("无。")
    for item in failures:
        reasons = "；".join(
            f"{failure['stage']}: {failure['reason']}" for failure in item["failures"]
        )
        lines.append(f"- `{item['case_id']}` {item['query']}：{reasons}")
    return "\n".join(lines) + "\n"


def run_evaluation() -> dict[str, Any]:
    tokenize("搜索推荐评测预热")
    cases = load_cases()
    qrels = load_qrels()
    run_id = f"eval-{uuid.uuid4().hex[:10]}"
    report = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "case_count": len(cases),
        "modes": {
            "fts": evaluate_mode("fts", cases, qrels),
            "hybrid": evaluate_mode("hybrid", cases, qrels),
        },
    }
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    json_path = settings.artifacts_dir / "eval_report.json"
    markdown_path = settings.artifacts_dir / "eval_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    with db_session() as connection:
        connection.execute(
            """
            INSERT INTO eval_runs(run_id, mode, metrics_json, report_path)
            VALUES(?, ?, ?, ?)
            """,
            (
                run_id,
                "baseline_vs_hybrid",
                json_dumps(
                    {
                        mode: payload["metrics"]
                        for mode, payload in report["modes"].items()
                    }
                ),
                str(json_path),
            ),
        )
    return report


def load_latest_report() -> dict[str, Any] | None:
    path = settings.artifacts_dir / "eval_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_policy_runs(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Adaptive-RAG sidecar for existing Replay results, never a self-judge.

    Optional per-turn expected_policy annotations contain required_chunk_ids,
    expected_action and hard_filter_violations. Missing semantic labels and
    provider usage remain null, rather than being counted as successes/zero.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        for turn in record.get("turns") or [record]:
            summaries = [item for item in turn.get("trace") or []
                         if item.get("node") == "after_sales.policy_followup_complete"]
            for summary in summaries:
                rows.append({"summary": summary, "turn": turn})
    if not rows:
        return {"cases": 0, "semantic_verified": False}
    latencies = sorted(float(row["summary"]["latency_ms"]) for row in rows)

    def mean(values: list[float]) -> float | None:
        return round(statistics.mean(values), 4) if values else None

    coverages, correct, multi, refusal_correct, first = [], [], [], [], []
    semantic, faithful, violations, costs, tokens, calls = [], [], [], [], [], []
    triggered = succeeded = repeated = 0
    for row in rows:
        summary, turn = row["summary"], row["turn"]
        trace = [item for item in turn.get("trace") or []
                 if item.get("task_id") == summary.get("task_id")]
        graders = [item for item in trace if item.get("node") == "after_sales.policy_evidence"]
        if graders:
            first.append(float(graders[0].get("next_action") == "answer"))
        answered = summary["stop_reason"] == "answer"
        extra = summary.get("extra_retrieval_calls", 0)
        triggered += int(extra > 0)
        succeeded += int(extra > 0 and answered)
        queries = [(item.get("topic"), item.get("query")) for item in trace
                   if item.get("node") == "after_sales.policy_followup"]
        repeated += len(queries) - len(set(queries))
        expected = turn.get("expected_policy") or {}
        required = set(expected.get("required_chunk_ids") or [])
        if required:
            coverage = len(required & set(summary.get("retrieved_chunk_ids") or summary.get("selected_chunk_ids") or [])) / len(required)
            coverages.append(coverage)
            if len(required) > 1:
                multi.append(float(answered and coverage == 1))
        if expected.get("expected_action"):
            actual = "answer" if answered else "ask_user" if summary["stop_reason"] == "ask_user" else "refuse"
            correct.append(float(actual == expected["expected_action"]))
            if actual == "refuse":
                refusal_correct.append(float(expected["expected_action"] == "refuse"))
        if "hard_filter_violations" in expected:
            violations.append(float(expected["hard_filter_violations"]))
        judge = turn.get("semantic_judge") or {}
        if judge.get("judge_mode") == "llm" and judge.get("evaluable") is not False:
            semantic.append(float(bool(judge.get("pass"))))
        if "grounding" in turn:
            faithful.append(float(bool(turn["grounding"].get("pass"))))
        if summary.get("llm_calls", 0) > 0:
            calls.append(float(summary["llm_calls"]))
            tokens.append(float(summary["llm_tokens"]))
        if summary.get("estimated_cost") is not None:
            costs.append(float(summary["estimated_cost"]))
    return {
        "cases": len(rows), "semantic_verified": bool(semantic),
        "evidence_coverage": mean(coverages), "action_correctness": mean(correct),
        "answer_correctness_faithfulness_judge": mean(semantic),
        "structural_grounding_pass_rate": mean(faithful),
        "answer_rate": mean([float(row["summary"]["stop_reason"] == "answer") for row in rows]),
        "refusal_precision": mean(refusal_correct), "multi_condition_success_rate": mean(multi),
        "first_pass_sufficient_rate": mean(first), "followup_trigger_rate": triggered / len(rows),
        "followup_success_rate": succeeded / triggered if triggered else None,
        "duplicate_followup_calls": repeated,
        "average_grader_invocations": mean([row["summary"]["assessment_calls"] for row in rows]),
        "average_llm_calls": mean(calls), "average_llm_tokens": mean(tokens),
        "average_estimated_cost": mean(costs), "hard_filter_violation_rate": mean(violations),
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": latencies[min(len(latencies) - 1, math.ceil(.95 * len(latencies)) - 1)],
    }


def policy_quality_gate(metrics: dict[str, Any], requirements: dict[str, Any]) -> dict[str, Any]:
    """Unknown measurements cannot pass the policy release gate."""
    failures, unavailable = [], []
    for key, threshold in requirements.items():
        if key == "independent_semantic_judge_required":
            if threshold and not metrics.get("semantic_verified"):
                unavailable.append("independent_semantic_judge")
            continue
        name, direction = key.rsplit("_", 1)
        value = metrics.get(name)
        if value is None:
            unavailable.append(name)
        elif (direction == "min" and value < threshold) or (direction == "max" and value > threshold):
            failures.append(name)
    return {"pass": not failures and not unavailable, "failures": failures, "unavailable": unavailable}


def main() -> None:
    report = run_evaluation()
    print(json.dumps(report["modes"], ensure_ascii=False, indent=2))
    print(f"report: {settings.artifacts_dir / 'eval_report.md'}")


if __name__ == "__main__":
    main()
