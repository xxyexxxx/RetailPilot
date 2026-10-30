from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RetailPilot.database import init_db
from RetailPilot.eval_dataset import seed_retrieval_eval_dataset


if __name__ == "__main__":
    init_db()
    print(json.dumps(seed_retrieval_eval_dataset(), ensure_ascii=False, indent=2))
