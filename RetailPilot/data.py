"""Database-backed data access.

The original in-memory product arrays were intentionally removed. Use the
catalog and database modules so every product, session, profile, and order is
persisted in SQLite.
"""

from .catalog import catalog_stats, get_product, get_products, list_brands, list_categories
from .database import catalog_ready, connect, init_db

__all__ = [
    "catalog_ready",
    "catalog_stats",
    "connect",
    "get_product",
    "get_products",
    "init_db",
    "list_brands",
    "list_categories",
]
