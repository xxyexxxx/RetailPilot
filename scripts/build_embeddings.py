from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RetailPilot.config import settings
from RetailPilot.embeddings import (
    index_metadata,
    model_slug,
    product_document,
    reset_embedding_index_cache,
    retrieval_prefixes,
)


def load_products() -> list[dict[str, str]]:
    absolute = settings.database_path.resolve().as_posix()
    connection = sqlite3.connect(f"file:{absolute}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT sku, title, brand, category, category_label, tags_text,
                   attributes_text, description
            FROM products
            ORDER BY sku
            """
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def build(model_name: str, batch_size: int, device: str, max_length: int) -> Path:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "缺少神经向量依赖，请先安装: "
            "python -m pip install -r requirements-embedding.txt"
        ) from exc

    products = load_products()
    if not products:
        raise SystemExit("商品库为空，请先运行 scripts/bootstrap_data.py")

    kwargs = {} if device == "auto" else {"device": device}
    model = SentenceTransformer(model_name, **kwargs)
    model.max_seq_length = min(int(model.max_seq_length), max_length)
    _, passage_prefix = retrieval_prefixes(model_name)
    documents = [f"{passage_prefix}{product_document(product)}" for product in products]
    matrix = model.encode(
        documents,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)

    target = settings.embedding_index_dir / model_slug(model_name)
    target.mkdir(parents=True, exist_ok=True)
    vector_tmp = target / "vectors.npy.tmp"
    sku_tmp = target / "skus.json.tmp"
    metadata_tmp = target / "metadata.json.tmp"

    with vector_tmp.open("wb") as stream:
        np.save(stream, matrix, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    sku_tmp.write_text(
        json.dumps([product["sku"] for product in products], ensure_ascii=False),
        encoding="utf-8",
    )
    metadata = index_metadata(model_name, int(matrix.shape[1]), len(products))
    metadata_tmp.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    vector_tmp.replace(target / "vectors.npy")
    sku_tmp.replace(target / "skus.json")
    metadata_tmp.replace(target / "metadata.json")
    reset_embedding_index_cache()
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an exact neural product-vector index")
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=settings.embedding_device)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    target = build(args.model, args.batch_size, args.device, args.max_length)
    print(f"embedding index: {target}")


if __name__ == "__main__":
    main()
