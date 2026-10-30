from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from RetailPilot.llm_config import invoke_judge, judge_config_status
from RetailPilot.utils import _parse_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make one real, secret-free LLM Judge connectivity probe"
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Only print the redacted configuration; do not call the endpoint",
    )
    args = parser.parse_args()
    status = judge_config_status()
    if not status.get("configured"):
        print(json.dumps(status, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    if args.config_only:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return

    started = time.perf_counter()
    try:
        content = invoke_judge(
            "只输出 JSON，不要 markdown："
            '{"ok":true,"kind":"llm_judge_probe"}'
        )
        parsed = _parse_json(content)
        if parsed.get("ok") is not True or parsed.get("kind") != "llm_judge_probe":
            raise ValueError("judge returned JSON that failed the probe contract")
    except Exception as exc:
        print(
            json.dumps(
                {
                    **status,
                    "probe": "failed",
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(1) from exc
    print(
        json.dumps(
            {
                **status,
                "probe": "passed",
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "response_contract": "valid_json",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
