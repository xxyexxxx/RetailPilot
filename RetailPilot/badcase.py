from __future__ import annotations

from typing import Any

from .evaluation import load_latest_report, run_evaluation


def scan() -> dict[str, Any]:
    report = run_evaluation()
    hybrid = report["modes"]["hybrid"]
    return {
        "run_id": report["run_id"],
        "badcases": [case for case in hybrid["cases"] if not case["pass"]],
        "root_causes": hybrid["root_causes"],
        "metrics": hybrid["metrics"],
    }


def latest() -> dict[str, Any] | None:
    return load_latest_report()
