"""Seed a small set of AI / programming books into the domestic catalogue.

Root-cause fix for the "推荐AI相关的书籍" badcase: the ShopSimulator import
contains no AI/CS books (its 164 books are exam-prep and imported originals),
so a correct parse still returns zero relevant hits. This script injects a
curated handful of domestic-style AI/CS books through the same
normalize_shopsimulator_item contract used by the real import, keeping
category=休闲娱乐文教 so existing category filtering and FTS bigram matching
work unchanged.

Run from the repo root:
    .\\.venv\\Scripts\\python.exe scripts/seed_domestic_ai_books.py
"""
from __future__ import annotations

import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from RetailPilot.catalog_adapters import normalize_shopsimulator_item
from RetailPilot.catalog_adapters import upsert_normalized_products

AI_BOOKS = [
    {
        "asin": "ai-book-0001",
        "title": "人工智能简史：从图灵测试到深度学习",
        "domain_zh": "休闲娱乐文教",
        "category": "书籍/杂志/报纸›计算机与互联网›人工智能",
        "attribute": ["正版图书", "人工智能", "科普读物", "深度学习", "图灵", "精装"],
        "shop_name": "人民邮电出版社官方旗舰店",
        "pricing": [59.8],
        "full_description": "人工智能入门科普书，介绍图灵测试、机器学习、深度学习的演进脉络。",
        "tag": "final",
    },
    {
        "asin": "ai-book-0002",
        "title": "机器学习实战：基于Python的经典算法实现",
        "domain_zh": "休闲娱乐文教",
        "category": "书籍/杂志/报纸›计算机与互联网›人工智能",
        "attribute": ["正版图书", "机器学习", "Python", "算法", "数据科学", "编程教材"],
        "shop_name": "清华大学出版社官方旗舰店",
        "pricing": [79.0],
        "full_description": "机器学习实战教程，用Python实现线性回归、决策树、聚类等经典算法。",
        "tag": "final",
    },
    {
        "asin": "ai-book-0003",
        "title": "深度学习入门：基于Python的理论与实现",
        "domain_zh": "休闲娱乐文教",
        "category": "书籍/杂志/报纸›计算机与互联网›人工智能",
        "attribute": ["正版图书", "深度学习", "神经网络", "Python", "编程教材"],
        "shop_name": "机械工业出版社官方旗舰店",
        "pricing": [69.9],
        "full_description": "深度学习入门书，从感知机到卷积神经网络，配Python代码示例。",
        "tag": "final",
    },
    {
        "asin": "ai-book-0004",
        "title": "动手学深度学习（PyTorch版）",
        "domain_zh": "休闲娱乐文教",
        "category": "书籍/杂志/报纸›计算机与互联网›人工智能",
        "attribute": ["正版图书", "深度学习", "PyTorch", "机器学习", "编程教材"],
        "shop_name": "电子工业出版社官方旗舰店",
        "pricing": [99.0],
        "full_description": "深度学习实战教材，覆盖神经网络、注意力机制、大模型训练基础。",
        "tag": "final",
    },
    {
        "asin": "ai-book-0005",
        "title": "Python编程：从入门到实践",
        "domain_zh": "休闲娱乐文教",
        "category": "书籍/杂志/报纸›计算机与互联网›编程",
        "attribute": ["正版图书", "Python", "编程入门", "程序设计", "计算机教材"],
        "shop_name": "博文视点官方旗舰店",
        "pricing": [59.0],
        "full_description": "Python编程入门经典教材，涵盖基础语法、数据结构和项目实战。",
        "tag": "final",
    },
    {
        "asin": "ai-book-0006",
        "title": "算法导论（第3版）计算机科学经典教材",
        "domain_zh": "休闲娱乐文教",
        "category": "书籍/杂志/报纸›计算机与互联网›编程",
        "attribute": ["正版图书", "算法", "数据结构", "计算机科学", "编程教材"],
        "shop_name": "中信出版官方旗舰店",
        "pricing": [128.0],
        "full_description": "计算机算法经典教材，覆盖排序、图算法、动态规划与NP完全性。",
        "tag": "final",
    },
    {
        "asin": "ai-book-0007",
        "title": "机器学习（周志华 著）",
        "domain_zh": "休闲娱乐文教",
        "category": "书籍/杂志/报纸›计算机与互联网›人工智能",
        "attribute": ["正版图书", "机器学习", "人工智能", "算法", "高校教材"],
        "shop_name": "图灵教育官方旗舰店",
        "pricing": [88.0],
        "full_description": "机器学习经典教材，系统讲解监督学习、无监督学习与集成学习。",
        "tag": "final",
    },
    {
        "asin": "ai-book-0008",
        "title": "大模型时代：生成式人工智能技术与应用",
        "domain_zh": "休闲娱乐文教",
        "category": "书籍/杂志/报纸›计算机与互联网›人工智能",
        "attribute": ["正版图书", "大模型", "生成式AI", "人工智能", "科技读物"],
        "shop_name": "北京大学出版社官方旗舰店",
        "pricing": [68.0],
        "full_description": "面向大众的生成式AI科普书，介绍大语言模型、图像生成与行业应用。",
        "tag": "final",
    },
    {
        "asin": "ai-book-0009",
        "title": "数据分析与挖掘：用Python做机器学习",
        "domain_zh": "休闲娱乐文教",
        "category": "书籍/杂志/报纸›计算机与互联网›人工智能",
        "attribute": ["正版图书", "数据分析", "Python", "机器学习", "数据挖掘"],
        "shop_name": "化学工业出版社官方旗舰店",
        "pricing": [72.0],
        "full_description": "数据分析与数据挖掘实战书，结合Python讲解特征工程与模型评估。",
        "tag": "final",
    },
    {
        "asin": "ai-book-0010",
        "title": "人工智能产品经理：从0到1实战指南",
        "domain_zh": "休闲娱乐文教",
        "category": "书籍/杂志/报纸›计算机与互联网›人工智能",
        "attribute": ["正版图书", "人工智能", "产品经理", "行业应用", "商业读物"],
        "shop_name": "异步图书官方旗舰店",
        "pricing": [64.9],
        "full_description": "面向产品经理的AI科普与应用书，讲解AI产品设计、落地与商业化。",
        "tag": "final",
    },
    {
        "asin": "ai-book-0011",
        "title": "程序员修炼之道：从小工到专家",
        "domain_zh": "休闲娱乐文教",
        "category": "书籍/杂志/报纸›计算机与互联网›编程",
        "attribute": ["正版图书", "编程", "程序员", "软件开发", "职业进阶"],
        "shop_name": "凤凰联动官方旗舰店",
        "pricing": [49.9],
        "full_description": "程序员进阶经典书，涵盖编码规范、重构、测试与工程素养。",
        "tag": "final",
    },
    {
        "asin": "ai-book-0012",
        "title": "神经网络与深度学习（邱锡鹏 著）",
        "domain_zh": "休闲娱乐文教",
        "category": "书籍/杂志/报纸›计算机与互联网›人工智能",
        "attribute": ["正版图书", "神经网络", "深度学习", "人工智能", "高校教材"],
        "shop_name": "智学图书官方旗舰店",
        "pricing": [89.0],
        "full_description": "神经网络与深度学习高校教材，从数学基础到CNN/RNN/Transformer。",
        "tag": "final",
    },
]


def main() -> None:
    products = [normalize_shopsimulator_item(item) for item in AI_BOOKS]
    inserted = upsert_normalized_products(products, replace=True)
    print(f"inserted {inserted}/{len(products)} AI/CS books")
    for product in products:
        print(f"  {product['sku']}  {product['title'][:44]}  ¥{product['price']}")


if __name__ == "__main__":
    main()
