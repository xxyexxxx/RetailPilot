from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
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

from RetailPilot.catalog import list_brands
from RetailPilot.catalog_adapters import iter_shopsimulator
from RetailPilot.config import settings
from RetailPilot.database import get_metadata
from RetailPilot.llm import maybe_refine_query
from RetailPilot.query import parse_query
from RetailPilot.retrieval import search_engine


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _load_tasks(split: str, limit: int, seed: int, source: Path) -> list[dict[str, Any]]:
    """Build tasks straight from the ShopSimulator source file.

    The `shopping_tasks` table was removed, so tasks are regenerated from the
    source catalog here instead of a database. `acceptable_skus` is set to the
    originating product sku so the recall metric is explicit (it is a lookup of
    the query's own source product, not a generalisation benchmark).
    """
    tasks: list[dict[str, Any]] = []
    for product in iter_shopsimulator(source):
        product_split = str(product["commerce"]["dataset_split"])
        if split and product_split != split:
            continue
        sku = product["sku"]
        for index, instruction in enumerate(product.get("instructions") or []):
            text = instruction.get("instruction") if isinstance(instruction, dict) else instruction
            if not text:
                continue
            tasks.append(
                {
                    "task_id": f"{sku}-{index}",
                    "query": text,
                    "hard_constraints": {},
                    "acceptable_skus": [sku],
                    "tags": [],
                }
            )
    random.Random(seed).shuffle(tasks)
    return tasks[:limit]


def _run_one(
    index: int,
    task: dict[str, Any],
    *,
    mode: str,
    session_id: str,
    top_k: int,
    brands: list[str],
) -> dict[str, Any]:
    try:
        parsed_query = maybe_refine_query(
            parse_query(task["query"], known_brands=brands),
            {"evaluation_task_id": task["task_id"]},
        )
        result = search_engine.search(
            parsed_query,
            user_id=f"batch-user-{session_id}",
            limit=top_k,
            mode=mode,
            session_id=session_id,
            known_brands=brands,
        )
    except Exception as exc:
        return {
            "sample_index": index,
            **task,
            "pass": False,
            "failure_types": ["system_error"],
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
        }

    products = result.get("products") or []
    result_skus = [product["sku"] for product in products]
    targets = set(task["acceptable_skus"])
    target_ranks = [
        rank for rank, sku in enumerate(result_skus, start=1) if sku in targets
    ]
    target_rank = min(target_ranks, default=None)
    parsed = result["parsed_query"]
    query_guard = any(
        step.get("node") == "query_guard" and step.get("blocked")
        for step in result.get("trace") or []
    )
    failures: list[str] = []
    if query_guard:
        failures.append("query_guard")
    if not products:
        failures.append("zero_result")
    if target_rank is None:
        failures.append("target_miss_at_10")
    return {
        "sample_index": index,
        **task,
        "request_id": result["request_id"],
        "pass": target_rank is not None,
        "failure_types": list(dict.fromkeys(failures)),
        "target_rank": target_rank,
        "result_count": len(products),
        "result_skus": result_skus,
        "top_products": [
            {
                "sku": product["sku"],
                "title": product["title"],
                "category": product["category"],
                "price": product["price"],
                "score": product.get("score"),
                "reason": product.get("reason", ""),
            }
            for product in products
        ],
        "parsed_query": parsed,
        "query_guard": query_guard,
        "trace": result.get("trace") or [],
        "latency_ms": float(result.get("latency_ms") or 0.0),
        "retrieval": result.get("retrieval") or {},
        "relaxations": result.get("relaxations") or [],
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if not record.get("error")]
    total = len(records)
    latencies = [float(record.get("latency_ms") or 0.0) for record in completed]
    retrieval_latencies = [
        float(record.get("latency_ms") or 0.0)
        for record in completed
        if any(step.get("node") == "structured_filter" for step in record.get("trace") or [])
    ]

    def rate(predicate: Any) -> float:
        return round(sum(bool(predicate(record)) for record in records) / max(1, total), 4)

    failure_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for record in records:
        for failure in record.get("failure_types") or []:
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
        category = str((record.get("parsed_query") or {}).get("category") or "<none>")
        category_counts[category] = category_counts.get(category, 0) + 1

    return {
        "tasks": total,
        "completed": len(completed),
        "system_errors": total - len(completed),
        "target_recall_at_1": rate(lambda record: record.get("target_rank") == 1),
        "target_recall_at_3": rate(
            lambda record: record.get("target_rank") is not None
            and int(record["target_rank"]) <= 3
        ),
        "target_recall_at_10": rate(lambda record: record.get("target_rank") is not None),
        "mean_reciprocal_rank": round(
            sum(1.0 / int(record["target_rank"]) for record in records if record.get("target_rank"))
            / max(1, total),
            4,
        ),
        "zero_result_rate": rate(lambda record: record.get("result_count", 0) == 0),
        "query_guard_rate": rate(lambda record: record.get("query_guard")),
        "category_extraction_rate": rate(
            lambda record: bool((record.get("parsed_query") or {}).get("category"))
        ),
        "product_term_extraction_rate": rate(
            lambda record: bool((record.get("parsed_query") or {}).get("grounded_target_terms"))
        ),
        "price_extraction_rate": rate(
            lambda record: (record.get("parsed_query") or {}).get("price_min") is not None
            or (record.get("parsed_query") or {}).get("price_max") is not None
        ),
        "latency_p50_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "latency_p95_ms": round(_percentile(latencies, 0.95), 2),
        "retrieval_entered": len(retrieval_latencies),
        "retrieval_latency_p50_ms": (
            round(statistics.median(retrieval_latencies), 2) if retrieval_latencies else 0.0
        ),
        "retrieval_latency_p95_ms": round(_percentile(retrieval_latencies, 0.95), 2),
        "failure_counts": dict(sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))),
        "parsed_category_counts": dict(
            sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


def _markdown(manifest: dict[str, Any], summary: dict[str, Any], badcases: list[dict[str, Any]]) -> str:
    lines = [
        f"# ShopSimulator Query Batch {manifest['run_id']}",
        "",
        f"- Split / 样本：`{manifest['split']}` / {summary['tasks']}",
        f"- Query 理解 / Embedding：`{manifest['intent_mode']}` / `{manifest['embedding_model']}`",
        f"- Target Recall@10：{summary['target_recall_at_10']:.2%}",
        f"- Zero Result Rate：{summary['zero_result_rate']:.2%}",
        f"- Query Guard Rate：{summary['query_guard_rate']:.2%}",
        f"- P50 / P95：{summary['latency_p50_ms']} / {summary['latency_p95_ms']} ms",
        f"- 进入召回：{summary['retrieval_entered']} / {summary['tasks']}；"
        f"召回 P50 / P95：{summary['retrieval_latency_p50_ms']} / "
        f"{summary['retrieval_latency_p95_ms']} ms",
        "",
        "## Failure 分布",
        "",
        "| Failure | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in summary["failure_counts"].items())
    lines.extend(["", "## 代表 Badcase", ""])
    for record in badcases[:20]:
        parsed = record.get("parsed_query") or {}
        lines.append(
            f"- `{record['task_id']}` {record['query']}  "
            f"失败：{', '.join(record.get('failure_types') or [])}；"
            f"类目：{parsed.get('category')}；检索词：{parsed.get('grounded_target_terms')}；"
            f"目标排名：{record.get('target_rank')}"
        )
    return "\n".join(lines) + "\n"


def _badcase_markdown(
    manifest: dict[str, Any], summary: dict[str, Any], badcases: list[dict[str, Any]]
) -> str:
    buckets: dict[str, list[dict[str, Any]]] = {
        "query_guard": [],
        "zero_result_after_retrieval": [],
        "target_miss_with_results": [],
        "system_error": [],
    }
    for record in badcases:
        if record.get("error"):
            buckets["system_error"].append(record)
        elif record.get("query_guard"):
            buckets["query_guard"].append(record)
        elif not record.get("result_count"):
            buckets["zero_result_after_retrieval"].append(record)
        else:
            buckets["target_miss_with_results"].append(record)

    lines = [
        f"# Badcase Analysis {manifest['run_id']}",
        "",
        "## 分层结论",
        "",
        "| 分层 | 数量 | 含义 |",
        "|---|---:|---|",
        f"| Query Guard | {len(buckets['query_guard'])} | Query 理解未建立可检索商品锚点 |",
        f"| 进入召回后空结果 | {len(buckets['zero_result_after_retrieval'])} | 结构过滤或候选门禁过严 |",
        f"| 有结果但目标未进 Top-K | {len(buckets['target_miss_with_results'])} | 召回/排序/标注口径问题 |",
        f"| 系统错误 | {len(buckets['system_error'])} | 代码或运行环境异常 |",
        "",
        "## Query 解析信号",
        "",
        f"- 类目提取率：{summary['category_extraction_rate']:.2%}",
        f"- 商品词提取率：{summary['product_term_extraction_rate']:.2%}",
        f"- 价格提取率：{summary['price_extraction_rate']:.2%}",
        "",
        "## 各层代表样本",
    ]
    for bucket, records in buckets.items():
        lines.extend(["", f"### {bucket} ({len(records)})", ""])
        for record in records[:10]:
            parsed = record.get("parsed_query") or {}
            top = (record.get("top_products") or [{}])[0]
            lines.append(
                f"- `{record.get('task_id')}` {record.get('query')}  "
                f"解析类目=`{parsed.get('category')}`，价格上限=`{parsed.get('price_max')}`，"
                f"Top1=`{top.get('sku', '-')}` {top.get('title', '')}"
            )
        if not records:
            lines.append("- 无")
    return "\n".join(lines) + "\n"


def _comparison_markdown(
    current_manifest: dict[str, Any], current: dict[str, Any], baseline_dir: Path
) -> str:
    baseline_summary = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
    baseline_manifest = json.loads((baseline_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = [
        ("Target Recall@1", "target_recall_at_1", True),
        ("Target Recall@3", "target_recall_at_3", True),
        ("Target Recall@10", "target_recall_at_10", True),
        ("MRR", "mean_reciprocal_rank", True),
        ("Zero Result Rate", "zero_result_rate", False),
        ("Query Guard Rate", "query_guard_rate", False),
        ("P95 (ms)", "latency_p95_ms", False),
    ]
    lines = [
        "# Baseline Comparison",
        "",
        f"- Baseline: `{baseline_manifest.get('run_id')}`",
        f"- Current: `{current_manifest.get('run_id')}`",
        "",
        "| Metric | Baseline | Current | Delta | Direction |",
        "|---|---:|---:|---:|---|",
    ]
    for label, key, higher_is_better in metrics:
        before = float(baseline_summary.get(key) or 0.0)
        after = float(current.get(key) or 0.0)
        delta = after - before
        improved = delta > 0 if higher_is_better else delta < 0
        direction = "improved" if improved else ("unchanged" if delta == 0 else "regressed")
        lines.append(f"| {label} | {before:.4f} | {after:.4f} | {delta:+.4f} | {direction} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible ShopSimulator real-query batch")
    parser.add_argument("--split", default="final", choices=["train", "final", "dev", ""])
    parser.add_argument("--source", type=Path, help="ShopSimulator catalog file (fine_items_eval_train_all.json[.gz])")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "fts"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--baseline-dir", type=Path,
        help="Optional previous run directory containing manifest.json and summary.json",
    )
    args = parser.parse_args()

    run_id = f"shopsim-{args.split}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    session_id = run_id
    output_dir = args.output_dir or settings.artifacts_dir / "real_query_runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    if not args.source:
        sys.exit("run_real_query_batch requires --source <shopsimulator-catalog>.json[.gz]")
    tasks = _load_tasks(args.split, max(1, args.limit), args.seed, args.source)
    brands = list_brands()

    records: list[dict[str, Any]] = []
    if args.workers <= 1:
        for index, task in enumerate(tasks):
            records.append(
                _run_one(index, task, mode=args.mode, session_id=session_id, top_k=args.top_k, brands=brands)
            )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _run_one,
                    index,
                    task,
                    mode=args.mode,
                    session_id=session_id,
                    top_k=args.top_k,
                    brands=brands,
                ): index
                for index, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                records.append(future.result())
    records.sort(key=lambda record: record["sample_index"])

    summary = _summary(records)
    first_retrieval = next((record.get("retrieval") for record in records if record.get("retrieval")), {}) or {}
    manifest = {
        "run_id": run_id,
        "session_id": session_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": args.split,
        "seed": args.seed,
        "limit": args.limit,
        "workers": args.workers,
        "top_k": args.top_k,
        "mode": args.mode,
        "database_path": str(settings.database_path.resolve()),
        "dataset_version": str(get_metadata("dataset_version", "")),
        "git_sha": str(get_metadata("git_sha", "")),
        "intent_mode": settings.intent_mode,
        "llm_mode": settings.llm_mode,
        "embedding_model": settings.embedding_model,
        "dense_backend": first_retrieval.get("dense_backend", ""),
        "embedding_index_version": first_retrieval.get("embedding_index_version", ""),
    }
    badcases = [record for record in records if not record.get("pass")]
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output_dir / "badcases.jsonl").open("w", encoding="utf-8") as stream:
        for record in badcases:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "summary.md").write_text(
        _markdown(manifest, summary, badcases), encoding="utf-8"
    )
    (output_dir / "badcase_analysis.md").write_text(
        _badcase_markdown(manifest, summary, badcases), encoding="utf-8"
    )
    if args.baseline_dir:
        (output_dir / "comparison.md").write_text(
            _comparison_markdown(manifest, summary, args.baseline_dir), encoding="utf-8"
        )
    print(json.dumps({"manifest": manifest, "summary": summary, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
