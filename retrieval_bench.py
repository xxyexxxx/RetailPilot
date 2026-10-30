from __future__ import annotations

import json

from RetailPilot.evaluation import run_evaluation


if __name__ == "__main__":
    report = run_evaluation()
    summary = {
        mode: payload["metrics"] for mode, payload in report["modes"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
