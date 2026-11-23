from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from RetailPilot.database import init_db
from RetailPilot.knowledge import sync_products
from RetailPilot.database import db_session
if __name__ == '__main__':
    init_db()
    with db_session() as db:
        db.execute('INSERT OR IGNORE INTO knowledge_dirty SELECT sku FROM products')
        sync_products(db)
    print('Canonical knowledge FTS is updated transactionally with each Unit.')
