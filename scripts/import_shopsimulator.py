from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RetailPilot.catalog_adapters import import_shopsimulator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a locally obtained ShopSimulator catalog without redistributing it."
    )
    parser.add_argument("source", type=Path, help="fine_items_eval_train_all.json[.gz]")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--no-tasks", action="store_true")
    args = parser.parse_args()
    report = import_shopsimulator(
        args.source,
        limit=args.limit,
        replace=args.replace,
        seed_tasks=not args.no_tasks,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
