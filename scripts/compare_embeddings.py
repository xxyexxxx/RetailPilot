from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RetailPilot.config import settings
from RetailPilot.embeddings import reset_embedding_index_cache
from RetailPilot.evaluation import evaluate_mode, load_cases, load_qrels


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare hash and neural dense recall")
    parser.add_argument("models", nargs="*", default=["hash", "BAAI/bge-small-zh-v1.5"])
    args = parser.parse_args()
    cases = load_cases()
    qrels = load_qrels()
    original_model = settings.embedding_model
    runs = {}
    try:
        for model in args.models:
            settings.embedding_model = model
            reset_embedding_index_cache()
            payload = evaluate_mode("hybrid", cases, qrels)
            runs[model] = {
                "metrics": payload["metrics"],
                "root_causes": payload["root_causes"],
                "badcase_ids": [case["case_id"] for case in payload["cases"] if not case["pass"]],
            }
    finally:
        settings.embedding_model = original_model
        reset_embedding_index_cache()

    report = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "case_count": len(cases),
        "qrel_count": sum(len(items) for items in qrels.values()),
        "runs": runs,
    }
    path = settings.artifacts_dir / "embedding_comparison.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {path}")


if __name__ == "__main__":
    main()
