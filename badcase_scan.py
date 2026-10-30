from __future__ import annotations

import json

from RetailPilot.evaluation import run_evaluation


if __name__ == "__main__":
    report = run_evaluation()
    hybrid = report["modes"]["hybrid"]
    badcases = [
        {
            "case_id": item["case_id"],
            "query": item["query"],
            "failures": item["failures"],
        }
        for item in hybrid["cases"]
        if not item["pass"]
    ]
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "badcase_count": len(badcases),
                "root_causes": hybrid["root_causes"],
                "badcases": badcases,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
