from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RetailPilot.database import init_db
from RetailPilot.judging import judge_pending_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge sampled search results offline")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    init_db()
    print(json.dumps(judge_pending_samples(args.limit), ensure_ascii=False, indent=2))
