from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sqlite3

from .config import settings
from .database import db_session, json_dumps


@dataclass(frozen=True, slots=True)
class QrelRule:
    category: str
    groups: tuple[tuple[str, ...], ...] = ()
    price_max: float | None = None
    negative_terms: tuple[str, ...] = ()
    expected: dict[str, Any] = field(default_factory=dict)


EXISTING_QREL_RULES: dict[str, QrelRule] = {
    "intent-electronics-budget": QrelRule(
        "Electronics", (("headphone", "headset", "earbud", "earphone"), ("wireless", "bluetooth")), 50
    ),
    "category-beauty-sensitive": QrelRule(
        "All Beauty", (("sensitive", "gentle", "hypoallergenic"), ("moistur", "hydrating", "cream")), 30
    ),
    "category-phone-accessories": QrelRule(
        "Cell Phones & Accessories", (("charger", "charging adapter"), ("fast charging", "quick charge", "usb c"))
    ),
    "category-sports": QrelRule(
        "Sports & Outdoors", (("waterproof", "water resistant"), ("lightweight", "light weight")), 100
    ),
    "negative-brand": QrelRule("Electronics", (), 200, ("apple",)),
    "complex-headphone-constraints": QrelRule(
        "Electronics",
        (("headphone", "headset"), ("wireless", "bluetooth"), ("noise cancel", "anc")),
        100,
        ("earbud", "earphone", "in-ear", "in ear"),
    ),
    "laptop-office-best": QrelRule("Computers", (("laptop", "notebook", "chromebook"),)),
    "correction-single-search": QrelRule("Computers", (("laptop", "notebook", "chromebook"),)),
}


ADDITIONAL_CASES: tuple[tuple[str, str, str, QrelRule, tuple[str, ...]], ...] = (
    # Electronics: paraphrase, cross-lingual concepts and form factors.
    ("semantic-headphone-commute", "通勤时想安静听歌的头戴式耳机", "通勤降噪耳机", QrelRule("Electronics", (("headphone", "headset"), ("noise cancel", "anc"))), ("semantic", "headphone")),
    ("semantic-speaker-wireless", "不用线连接手机的蓝牙音箱", "无线音箱", QrelRule("Electronics", (("speaker",), ("wireless", "bluetooth"))), ("semantic", "speaker")),
    ("semantic-camera-portable", "适合旅行拍视频的便携相机", "旅行相机", QrelRule("Electronics", (("camera",), ("portable", "compact", "travel"))), ("semantic", "camera")),
    ("semantic-microphone-meeting", "推荐数码产品里适合电脑开会的麦克风", "会议麦克风", QrelRule("Electronics", (("microphone", "mic"), ("computer", "usb", "meeting"))), ("semantic", "meeting")),
    ("semantic-watch-fitness", "运动时记录健康数据的智能手表", "运动智能手表", QrelRule("Electronics", (("smartwatch", "smart watch"), ("fitness", "health", "sport"))), ("semantic", "watch")),
    ("semantic-headphone-overear", "不要塞进耳朵里的蓝牙耳机", "非入耳蓝牙耳机", QrelRule("Electronics", (("headphone", "headset"), ("bluetooth", "wireless")), negative_terms=("earbud", "earphone", "in-ear", "in ear")), ("semantic", "negative")),
    # Computers.
    ("semantic-laptop-windows", "办公用的 Windows 笔记本电脑", "Windows 办公本", QrelRule("Computers", (("windows",), ("business", "office", "productivity"))), ("semantic", "computer")),
    ("semantic-laptop-intel", "想要英特尔处理器的笔记本电脑", "Intel 笔记本", QrelRule("Computers", (("intel",),)), ("semantic", "computer")),
    ("semantic-laptop-touch", "可以直接用手点屏幕的笔记本电脑", "触控笔记本", QrelRule("Computers", (("touchscreen", "touch screen"),)), ("semantic", "paraphrase")),
    ("semantic-laptop-gaming", "能玩大型游戏的笔记本电脑", "游戏本", QrelRule("Computers", (("gaming", "game"),)), ("semantic", "computer")),
    ("semantic-laptop-chromebook", "推荐 Chromebook 笔记本电脑", "Chromebook", QrelRule("Computers", (("chromebook", "chrome os"),)), ("semantic", "computer")),
    ("semantic-laptop-light", "经常出差背着不累的轻薄笔记本电脑", "轻薄便携本", QrelRule("Computers", (("lightweight", "thin", "portable"),)), ("semantic", "paraphrase")),
    # Cell-phone accessories.
    ("semantic-phone-fast-charge", "给手机快速补电的充电器", "手机快充", QrelRule("Cell Phones & Accessories", (("charger", "charging"), ("fast", "quick charge", "usb c"))), ("semantic", "phone")),
    ("semantic-phone-power-bank", "出门给手机续命的充电宝", "便携充电宝", QrelRule("Cell Phones & Accessories", (("power bank", "portable charger"),)), ("semantic", "phone")),
    ("semantic-phone-cable", "耐用的手机数据线", "手机数据线", QrelRule("Cell Phones & Accessories", (("cable", "cord"),)), ("semantic", "phone")),
    ("semantic-phone-protector", "防止屏幕刮花的手机配件", "手机屏幕保护", QrelRule("Cell Phones & Accessories", (("screen protector", "tempered glass"),)), ("semantic", "paraphrase")),
    ("semantic-phone-car-charge", "开车时给手机充电的手机配件", "车载充电", QrelRule("Cell Phones & Accessories", (("car charger", "vehicle charger", "car charging"),)), ("semantic", "scenario")),
    ("semantic-phone-wireless-charge", "不用插线就能充电的手机配件", "无线充电", QrelRule("Cell Phones & Accessories", (("wireless charg", "charging pad"),)), ("semantic", "paraphrase")),
    # Beauty.
    ("semantic-beauty-moisture", "冬天脸干用的保湿护肤品", "干燥肌保湿", QrelRule("All Beauty", (("moistur", "hydrating", "cream"),)), ("semantic", "beauty")),
    ("semantic-beauty-serum", "质地轻薄的精华护肤品", "护肤精华", QrelRule("All Beauty", (("serum", "essence"),)), ("semantic", "beauty")),
    ("semantic-beauty-sensitive", "容易泛红的人用的温和护肤品", "敏感肌温和护肤", QrelRule("All Beauty", (("sensitive", "gentle", "hypoallergenic"),)), ("semantic", "paraphrase")),
    ("semantic-beauty-perfume", "闻起来清新的香水", "清新香水", QrelRule("All Beauty", (("perfume", "fragrance", "eau de"),)), ("semantic", "beauty")),
    ("semantic-beauty-brush", "上妆用的美妆刷子", "化妆刷", QrelRule("All Beauty", (("brush", "applicator"),)), ("semantic", "beauty")),
    ("semantic-beauty-lip", "嘴唇干裂时用的美妆产品", "唇部护理", QrelRule("All Beauty", (("lip", "balm"),)), ("semantic", "paraphrase")),
    # Sports and outdoors.
    ("semantic-sports-waterproof", "下雨天也能用的运动装备", "防水运动装备", QrelRule("Sports & Outdoors", (("waterproof", "water resistant"),)), ("semantic", "sports")),
    ("semantic-sports-yoga", "在家拉伸用的运动装备", "瑜伽拉伸装备", QrelRule("Sports & Outdoors", (("yoga", "stretch"),)), ("semantic", "scenario")),
    ("semantic-sports-camping", "周末露营要带的户外用品", "露营用品", QrelRule("Sports & Outdoors", (("camp", "tent", "outdoor"),)), ("semantic", "sports")),
    ("semantic-sports-bottle", "跑步时方便喝水的运动装备", "运动水壶", QrelRule("Sports & Outdoors", (("bottle", "hydration"),)), ("semantic", "scenario")),
    ("semantic-sports-bike", "骑行时使用的运动装备", "骑行装备", QrelRule("Sports & Outdoors", (("bike", "bicycle", "cycling"),)), ("semantic", "sports")),
    ("semantic-sports-light", "长距离徒步背着轻松的户外用品", "轻量徒步装备", QrelRule("Sports & Outdoors", (("lightweight", "ultralight", "light weight"),)), ("semantic", "paraphrase")),
    # Appliances.
    ("semantic-appliance-coffee", "早上自动做咖啡的家电", "咖啡家电", QrelRule("Appliances", (("coffee", "espresso"),)), ("semantic", "appliance")),
    ("semantic-appliance-vacuum", "帮我清理地面灰尘的家电", "吸尘清洁家电", QrelRule("Appliances", (("vacuum", "floor cleaner"),)), ("semantic", "paraphrase")),
    ("semantic-appliance-heater", "冬天让房间暖和的家电", "取暖家电", QrelRule("Appliances", (("heater", "heating"),)), ("semantic", "appliance")),
    ("semantic-appliance-fan", "夏天让空气流动的家电", "风扇家电", QrelRule("Appliances", (("fan", "air circulator"),)), ("semantic", "paraphrase")),
    ("semantic-appliance-clean", "方便打扫厨房的家电", "厨房清洁家电", QrelRule("Appliances", (("clean", "wash"),)), ("semantic", "scenario")),
    # Video games.
    ("semantic-game-nintendo", "任天堂主机用的游戏配件", "Nintendo 游戏配件", QrelRule("Video Games", (("nintendo", "switch"),)), ("semantic", "gaming")),
    ("semantic-game-playstation", "PlayStation 玩家用的游戏配件", "PlayStation 配件", QrelRule("Video Games", (("playstation", "ps4", "ps5"),)), ("semantic", "gaming")),
    ("semantic-game-xbox", "微软主机用的游戏配件", "Xbox 配件", QrelRule("Video Games", (("xbox",),)), ("semantic", "gaming")),
    ("semantic-game-headset", "和队友语音沟通的游戏配件", "游戏耳麦", QrelRule("Video Games", (("headset", "headphone", "microphone"),)), ("semantic", "paraphrase")),
    ("semantic-game-wireless", "不用线连接的游戏手柄", "无线手柄", QrelRule("Video Games", (("gamepad", "controller"), ("wireless", "bluetooth"))), ("semantic", "gaming")),
)


def _content(product: dict[str, Any]) -> str:
    return " ".join(
        [product["title"], product["description"], product["attributes_text"]]
    ).casefold()


def _relevance(product: dict[str, Any], rule: QrelRule) -> int:
    if product["category"] != rule.category:
        return 0
    if rule.price_max is not None and float(product["price"]) > rule.price_max:
        return 0
    content = _content(product)
    if any(term.casefold() in content for term in rule.negative_terms):
        return 0
    if not rule.groups:
        return 2
    matched = sum(
        1 for group in rule.groups if any(term.casefold() in content for term in group)
    )
    if matched == len(rule.groups):
        return 3
    if matched >= max(1, (len(rule.groups) + 1) // 2):
        return 2
    return 0


def seed_retrieval_eval_dataset() -> dict[str, int]:
    rules = dict(EXISTING_QREL_RULES)
    with db_session() as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO eval_cases(
                case_id, name, query, context_json, expected_json, tags_json, enabled
            ) VALUES(?, ?, ?, '{}', ?, ?, 1)
            """,
            [
                (
                    case_id,
                    name,
                    query,
                    json_dumps(
                        {
                            "intent": "search",
                            "category": rule.category,
                            "min_results": 1,
                            **rule.expected,
                        }
                    ),
                    json_dumps(list(tags)),
                )
                for case_id, query, name, rule, tags in ADDITIONAL_CASES
            ],
        )
        rules.update({case_id: rule for case_id, _, _, rule, _ in ADDITIONAL_CASES})
        placeholders = ",".join("?" for _ in rules)
        connection.execute(
            f"DELETE FROM eval_qrels WHERE case_id IN ({placeholders})", list(rules)
        )
        product_rows = connection.execute(
            """
            SELECT sku, title, category, description, attributes_text, price,
                   quality_score, popularity_score
            FROM products
            """
        ).fetchall()
        products = [dict(row) for row in product_rows]
        qrel_rows: list[tuple[str, str, int, str, str]] = []
        populated = 0
        for case_id, rule in rules.items():
            judged = [
                (product, _relevance(product, rule)) for product in products
            ]
            relevant = [(product, relevance) for product, relevance in judged if relevance >= 2]
            relevant.sort(
                key=lambda item: (
                    item[1],
                    float(item[0]["quality_score"]),
                    float(item[0]["popularity_score"]),
                ),
                reverse=True,
            )
            if relevant:
                populated += 1
            qrel_rows.extend(
                (case_id, product["sku"], relevance, "heuristic:v1", "")
                for product, relevance in relevant[:200]
            )
        connection.executemany(
            """
            INSERT INTO eval_qrels(case_id, sku, relevance, source, note)
            VALUES(?, ?, ?, ?, ?)
            """,
            qrel_rows,
        )
    _write_holdout_db(qrel_rows)
    return {"cases": len(rules), "populated": populated, "qrels": len(qrel_rows)}


def _write_holdout_db(qrel_rows: list[tuple[str, str, int, str, str]]) -> None:
    """Keep graded relevance physically outside the indexed product database.

    The rows are still generated by the explicit heuristic rules in this
    module, not by asking the retrieval system to grade itself.  They are
    therefore a reproducible holdout artifact, not an independent human label.
    """

    settings.eval_holdout_path.parent.mkdir(parents=True, exist_ok=True)
    if not qrel_rows:
        return
    with sqlite3.connect(settings.eval_holdout_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_qrels (
                case_id TEXT NOT NULL,
                sku TEXT NOT NULL,
                relevance INTEGER NOT NULL CHECK(relevance IN (0, 1, 2, 3)),
                source TEXT NOT NULL DEFAULT 'heuristic:v1',
                note TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (case_id, sku)
            )
            """
        )
        placeholders = ",".join("?" for _ in {(row[0], row[1]) for row in qrel_rows})
        connection.execute(
            f"DELETE FROM eval_qrels WHERE (case_id || ':' || sku) IN ({placeholders})",
            [f"{row[0]}:{row[1]}" for row in qrel_rows],
        )
        connection.executemany(
            """
            INSERT INTO eval_qrels(case_id, sku, relevance, source, note)
            VALUES(?, ?, ?, ?, ?)
            """,
            qrel_rows,
        )
