from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from RetailPilot.database import db_session, init_db


def rebuild() -> dict[str, int]:
    init_db()
    with db_session() as connection:
        connection.execute("DELETE FROM product_fts")
        connection.execute(
            """
            INSERT INTO product_fts(sku, title, brand, category, tags, description, attributes)
            SELECT sku, title, brand, category || ' ' || category_label,
                   tags_text, SUBSTR(description, 1, 800), SUBSTR(attributes_text, 1, 2400)
            FROM products
            """
        )
        count = connection.execute("SELECT COUNT(*) FROM product_fts").fetchone()[0]
        connection.execute("INSERT INTO product_fts(product_fts) VALUES('optimize')")
    return {"indexed_products": count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the bounded product FTS index")
    parser.parse_args()
    print(json.dumps(rebuild(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
