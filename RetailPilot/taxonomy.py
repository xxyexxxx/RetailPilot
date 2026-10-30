from __future__ import annotations

import re
from collections import OrderedDict

_ASCII_ALIAS_PATTERNS: dict[str, re.Pattern] = {}


# 高频电商同音/形似错别字 -> 标准检索词。只收录多字整词（不做单字替换），
# 避免"一只鸡"被误改成"一只机"。长词优先，防止"无线饵机"先被"饵机"拆开替换。
CORRECTION_DICT: "OrderedDict[str, str]" = OrderedDict(
    sorted(
        {
            "无线饵机": "无线耳机",
            "蓝牙饵机": "蓝牙耳机",
            "降噪饵机": "降噪耳机",
            "电竞饵机": "电竞耳机",
            "无线手几": "无线手机",
            "蓝牙手几": "蓝牙手机",
            "笔记笨": "笔记本",
            "平板电恼": "平板电脑",
            "充电汽": "充电器",
            "充点器": "充电器",
            "充电葆": "充电宝",
            "洗面乃": "洗面奶",
            "钢化莫": "钢化膜",
            "钢化摸": "钢化膜",
            "手机贴莫": "手机贴膜",
            "保护摸": "保护膜",
            "鼠标垫儿": "鼠标垫",
            "数据县": "数据线",
            "网县": "网线",
            "鼠票": "鼠标",
            "护夫品": "护肤品",
            "化妆拼": "化妆品",
            "蓝芽": "蓝牙",
            "手鸡": "手机",
            "手几": "手机",
            "饵机": "耳机",
            "儿机": "耳机",
            "耳几": "耳机",
            "面模": "面膜",
        }.items(),
        key=lambda item: -len(item[0]),
    )
)


def correction_candidates(raw_query: str) -> list[str]:
    """Generate corrected retrieval candidates for typo-laden queries.

    Homophone/shape-similar typos (手鸡 -> 手机) are replaced using the
    correction dictionary. The corrected text is returned as a retrieval-only
    candidate; the raw query and source spans are left untouched so the
    grounding gates still enforce verbatim provenance.
    """
    text = str(raw_query or "").strip()
    if not text:
        return []
    corrected = text
    for wrong, right in CORRECTION_DICT.items():
        if wrong in corrected:
            corrected = corrected.replace(wrong, right)
    if corrected == text:
        return []
    return [corrected]


def alias_in_text(text: str, alias: str) -> bool:
    """Match an alias inside user text.

    ASCII aliases (e.g. "AI", "Python") match on whole-word boundaries so
    "AI" never fires inside "email"/"said"/"detail". CJK aliases keep plain
    substring matching because Chinese has no word delimiters.
    """
    if alias.isascii() and alias.isalnum():
        pattern = _ASCII_ALIAS_PATTERNS.get(alias)
        if pattern is None:
            pattern = re.compile(
                rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
                re.IGNORECASE,
            )
            _ASCII_ALIAS_PATTERNS[alias] = pattern
        return pattern.search(text) is not None
    return alias.lower() in text.lower()


CATEGORY_LABELS = {
    "All Beauty": "美妆个护",
    "Appliances": "家用电器",
    "Cell Phones & Accessories": "手机与配件",
    "Computers": "电脑办公",
    "Electronics": "数码电子",
    "Sports & Outdoors": "运动户外",
    "Video Games": "游戏",
    "家居家装": "家居家装",
    "服饰鞋包饰品": "服饰鞋包饰品",
    "休闲娱乐文教": "休闲娱乐文教",
    "美妆个护健康": "美妆个护健康",
    "生产材料农用品": "生产材料农用品",
    "家用电器数码": "家用电器数码",
    "运动户外交通": "运动户外交通",
    "食品饮品": "食品饮品",
    "母婴儿童": "母婴儿童",
}

# Longer aliases must win before shorter ones.
CATEGORY_ALIASES = OrderedDict(
    sorted(
        {
            "家居家装": "家居家装",
            "家具": "家居家装",
            "枕头": "家居家装",
            "床垫": "家居家装",
            "服饰鞋包饰品": "服饰鞋包饰品",
            "衣服": "服饰鞋包饰品",
            "鞋子": "服饰鞋包饰品",
            "休闲娱乐文教": "休闲娱乐文教",
            "图书": "休闲娱乐文教",
            "书籍": "休闲娱乐文教",
            "书本": "休闲娱乐文教",
            "书": "休闲娱乐文教",
            "文具": "休闲娱乐文教",
            "美妆个护健康": "美妆个护健康",
            "生产材料农用品": "生产材料农用品",
            "农用品": "生产材料农用品",
            "家用电器数码": "家用电器数码",
            "运动户外交通": "运动户外交通",
            "食品饮品": "食品饮品",
            "零食": "食品饮品",
            "饮料": "食品饮品",
            "吃的": "食品饮品",
            "喝的": "食品饮品",
            "水果": "食品饮品",
            "母婴儿童": "母婴儿童",
            "母婴用品": "母婴儿童",
            # 中文电子/数码/3C类词一律映射到中文类目「家用电器数码」，
            # 不再指向英文类目（Computers/Electronics/Cell Phones 等旧英文数据体系）。
            "笔记本电脑": "家用电器数码",
            "办公笔记本": "家用电器数码",
            "办公电脑": "家用电器数码",
            "电脑": "家用电器数码",
            "notebook computer": "家用电器数码",
            "chromebook": "家用电器数码",
            "macbook": "家用电器数码",
            "laptop": "家用电器数码",
            "notebook": "家用电器数码",
            "轻薄本": "家用电器数码",
            "游戏本": "家用电器数码",
            "笔记本": "家用电器数码",
            "机械键盘": "家用电器数码",
            "无线键盘": "家用电器数码",
            "游戏键盘": "家用电器数码",
            "无线鼠标": "家用电器数码",
            "游戏鼠标": "家用电器数码",
            "键盘": "家用电器数码",
            "鼠标": "家用电器数码",
            "手机配件": "家用电器数码",
            "手机壳": "家用电器数码",
            "充电器": "家用电器数码",
            "充电宝": "家用电器数码",
            "数据线": "家用电器数码",
            "蓝牙耳机": "家用电器数码",
            "蓝牙音箱": "家用电器数码",
            "降噪耳机": "家用电器数码",
            "智能手表": "家用电器数码",
            "数码产品": "家用电器数码",
            "电子产品": "家用电器数码",
            "手机": "家用电器数码",
            "耳机": "家用电器数码",
            "手表": "家用电器数码",
            "音箱": "家用电器数码",
            "相机": "家用电器数码",
            "家电": "家用电器数码",
            "电器": "家用电器数码",
            "游戏手柄": "休闲娱乐文教",
            "游戏配件": "休闲娱乐文教",
            "游戏": "休闲娱乐文教",
            "运动装备": "运动户外交通",
            "户外用品": "运动户外交通",
            "运动": "运动户外交通",
            "户外": "运动户外交通",
            "护肤品": "美妆个护健康",
            "化妆品": "美妆个护健康",
            "美容": "美妆个护健康",
            "香水": "美妆个护健康",
            "美妆": "美妆个护健康",
            "护肤": "美妆个护健康",
            "beauty": "美妆个护健康",
            "appliance": "家用电器数码",
            "phone": "家用电器数码",
            "electronics": "家用电器数码",
            "sports": "运动户外交通",
            "video game": "休闲娱乐文教",
        }.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)

CATEGORY_SEARCH_TERMS = {
    "All Beauty": ["beauty", "skin", "makeup", "hair", "fragrance", "美妆", "护肤"],
    "Appliances": ["appliance", "kitchen", "home", "家电", "电器"],
    "Cell Phones & Accessories": ["phone accessory", "mobile accessory", "手机", "配件"],
    "Computers": [
        "laptop",
        "notebook computer",
        "chromebook",
        "macbook",
        "computer",
        "笔记本电脑",
        "电脑办公",
    ],
    "Electronics": ["electronics", "electronic", "电子", "数码"],
    "Sports & Outdoors": ["sports", "outdoor", "fitness", "running", "运动", "户外"],
    "Video Games": ["video game", "gaming", "controller", "console", "游戏"],
    "家居家装": ["家居", "家具", "家装"],
    "服饰鞋包饰品": ["服饰", "鞋包", "饰品"],
    "休闲娱乐文教": ["休闲", "娱乐", "文教", "图书", "文具"],
    "美妆个护健康": ["美妆", "个护", "健康"],
    "生产材料农用品": ["生产材料", "农用品"],
    "家用电器数码": ["家电", "数码", "电器"],
    "运动户外交通": ["运动", "户外", "交通"],
    "食品饮品": ["食品", "饮品", "零食"],
    "母婴儿童": ["母婴", "儿童", "婴儿"],
}

ALIAS_SEARCH_TERMS = {
    "书籍": ["书籍", "图书", "教材", "book"],
    "书本": ["书籍", "图书", "教材", "book"],
    "笔记本电脑": ["laptop", "notebook computer", "chromebook", "macbook"],
    "办公笔记本": ["business laptop", "office laptop", "notebook computer"],
    "办公电脑": ["business laptop", "office laptop", "notebook computer"],
    "笔记本": ["laptop", "notebook computer", "chromebook", "macbook"],
    "轻薄本": ["thin and light laptop", "ultrabook", "portable laptop"],
    "游戏本": ["gaming laptop", "notebook computer"],
    "机械键盘": ["mechanical keyboard", "gaming keyboard"],
    "无线键盘": ["wireless keyboard", "bluetooth keyboard"],
    "游戏键盘": ["gaming keyboard", "mechanical keyboard"],
    "键盘": ["keyboard", "keypad"],
    "无线鼠标": ["wireless mouse", "bluetooth mouse"],
    "游戏鼠标": ["gaming mouse", "rgb mouse"],
    "鼠标": ["mouse", "mice"],
    "laptop": ["laptop", "notebook computer", "chromebook", "macbook"],
    "notebook": ["laptop", "notebook computer"],
    "notebook computer": ["laptop", "notebook computer"],
    "chromebook": ["chromebook", "laptop"],
    "macbook": ["macbook", "laptop"],
    "蓝牙耳机": ["bluetooth headphone", "wireless earbuds", "earphone"],
    "蓝牙音箱": ["bluetooth speaker", "wireless speaker"],
    "降噪耳机": ["noise cancelling headphone", "anc earbuds"],
    "耳机": ["headphone", "earbuds", "earphone", "headset"],
    "智能手表": ["smart watch", "wearable"],
    "手表": ["watch", "wearable"],
    "音箱": ["speaker", "bluetooth speaker"],
    "相机": ["camera", "photography"],
    "手机壳": ["phone case", "protective case"],
    "充电器": ["phone charger", "charging adapter"],
    "充电宝": ["power bank", "portable charger"],
    "数据线": ["charging cable", "usb cable"],
    "手机配件": ["phone accessory", "mobile accessory"],
    "护肤品": ["skin care", "skincare"],
    "护肤": ["skin care", "skincare"],
    "香水": ["fragrance", "perfume"],
    "美妆": ["beauty", "makeup"],
    "游戏手柄": ["game controller", "gamepad"],
    "游戏配件": ["gaming accessory"],
    "运动装备": ["sports gear", "fitness equipment"],
    "户外用品": ["outdoor gear", "camping"],
}

PRODUCT_ACCESSORY_EXCLUSIONS = {
    "笔记本电脑": [
        "case",
        "bag",
        "sleeve",
        "cover",
        "stand",
        "holder",
        "replacement",
        "charger",
        "adapter",
        "docking station",
        "dock for",
        "cooling pad",
        "screen protector",
        "keyboard skin",
        "decal",
        "sticker",
    ],
    "办公笔记本": [
        "case",
        "bag",
        "sleeve",
        "cover",
        "stand",
        "replacement",
        "charger",
        "adapter",
        "docking station",
        "cooling pad",
        "screen protector",
    ],
    "办公电脑": [
        "case",
        "bag",
        "sleeve",
        "cover",
        "stand",
        "replacement",
        "charger",
        "adapter",
        "docking station",
        "cooling pad",
        "screen protector",
    ],
    "笔记本": [
        "case",
        "bag",
        "sleeve",
        "cover",
        "stand",
        "holder",
        "replacement",
        "charger",
        "adapter",
        "docking station",
        "dock for",
        "cooling pad",
        "screen protector",
        "keyboard skin",
        "decal",
        "sticker",
    ],
    "轻薄本": [
        "case",
        "bag",
        "sleeve",
        "cover",
        "stand",
        "charger",
        "adapter",
        "dock",
    ],
    "游戏本": [
        "case",
        "bag",
        "sleeve",
        "cover",
        "stand",
        "charger",
        "adapter",
        "cooling pad",
    ],
    "laptop": [
        "case",
        "bag",
        "sleeve",
        "cover",
        "stand",
        "holder",
        "replacement",
        "charger",
        "adapter",
        "docking station",
        "dock for",
        "cooling pad",
        "screen protector",
        "keyboard skin",
        "decal",
        "sticker",
    ],
    "notebook": [
        "case",
        "bag",
        "sleeve",
        "cover",
        "stand",
        "replacement",
        "charger",
        "adapter",
        "dock",
    ],
    "notebook computer": [
        "case",
        "bag",
        "sleeve",
        "cover",
        "stand",
        "replacement",
        "charger",
        "adapter",
        "dock",
    ],
    "chromebook": [
        "case",
        "bag",
        "sleeve",
        "cover",
        "charger",
        "adapter",
        "screen protector",
    ],
    "macbook": [
        "case",
        "bag",
        "sleeve",
        "cover",
        "charger",
        "adapter",
        "dock",
        "screen protector",
    ],
    "蓝牙耳机": [
        "replacement",
        "compatible with",
        "ear pad",
        "earpad",
        "cushion",
        "adapter",
        "case for",
        "ear hook",
        "earhook",
        "cable for",
        "headphone jack",
    ],
    "降噪耳机": [
        "replacement",
        "compatible with",
        "ear pad",
        "earpad",
        "cushion",
        "adapter",
        "case for",
        "ear hook",
        "earhook",
        "cable for",
        "headphone jack",
    ],
    "耳机": [
        "replacement",
        "compatible with",
        "ear pad",
        "earpad",
        "cushion",
        "adapter",
        "case for",
        "ear hook",
        "earhook",
        "cable for",
        "headphone jack",
    ],
    "智能手表": ["watch band", "watch strap", "screen protector", "case for"],
    "相机": ["camera case", "camera bag", "lens cap", "screen protector"],
    "游戏手柄": ["controller case", "controller stand", "replacement"],
}

# Query concept expansion doubles as a lightweight cross-lingual semantic layer.
CONCEPT_ALIASES = OrderedDict(
    sorted(
        {
            "主动降噪": ["noise canceling", "noise cancelling", "active noise cancellation", "anc"],
            "降噪": ["noise canceling", "noise cancelling", "anc"],
            "续航": ["battery life", "long battery", "rechargeable"],
            "长续航": ["long battery life", "rechargeable", "battery"],
            "防水": ["waterproof", "water resistant", "ipx"],
            "防汗": ["sweat resistant", "water resistant", "sports"],
            "轻便": ["lightweight", "portable", "compact"],
            "便携": ["portable", "compact", "travel"],
            "跑步": ["running", "sports", "fitness"],
            "健身": ["fitness", "workout", "sports"],
            "通勤": ["commute", "portable", "wireless"],
            "办公": ["office", "business", "productivity", "work"],
            "商务": ["business", "office", "professional", "work"],
            "旅行": ["travel", "portable", "compact"],
            "敏感肌": ["sensitive skin", "hypoallergenic", "fragrance free"],
            "保湿": ["moisturizing", "hydrating", "moisture"],
            "控油": ["oil control", "matte", "cleanser"],
            "防晒": ["sunscreen", "sun protection", "spf"],
            "天然": ["natural", "plant based"],
            "有机": ["organic", "natural"],
            "无香": ["fragrance free", "unscented"],
            "送礼": ["gift", "giftable", "present"],
            "儿童": ["kids", "children", "child"],
            "老人": ["senior", "easy to use", "large display"],
            "性价比": ["value", "affordable", "popular"],
            "高评分": ["top rated", "high rating"],
            "无线": ["wireless", "bluetooth"],
            "入耳式": ["in ear", "in-ear", "earbud", "earbuds", "earphone"],
            "耳塞式": ["in ear", "in-ear", "earbud", "earbuds", "earphone"],
            "头戴式": ["over ear", "over-ear", "on ear", "on-ear", "headphone"],
            "快充": ["fast charging", "quick charge", "usb c"],
            "苹果": ["apple", "iphone", "ipad"],
            "安卓": ["android", "usb c"],
            "电竞": ["gaming", "low latency", "rgb"],
            "静音": ["quiet", "low noise"],
            "省电": ["energy efficient", "low power"],
            "AI": ["人工智能", "机器学习", "深度学习", "artificial intelligence", "machine learning", "deep learning", "ai"],
            "人工智能": ["artificial intelligence", "machine learning", "deep learning", "ai", "neural network"],
            "机器学习": ["machine learning", "artificial intelligence", "deep learning", "ai"],
            "深度学习": ["deep learning", "neural network", "machine learning", "ai"],
            "神经网络": ["neural network", "deep learning", "machine learning", "ai"],
            "编程": ["programming", "coding", "software development", "python", "developer"],
            "程序员": ["programming", "coding", "developer", "software"],
            "Python": ["python", "programming", "coding"],
        }.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)

MUST_HAVE_CONCEPT_ALIASES = {
    "主动降噪",
    "降噪",
    "无线",
    "防水",
    "防汗",
    "快充",
    "无香",
    "入耳式",
    "耳塞式",
    "头戴式",
}

SORT_ALIASES = {
    "最好的": "rating_desc",
    "最好": "rating_desc",
    "最优": "rating_desc",
    "最便宜": "price_asc",
    "便宜优先": "price_asc",
    "价格低": "price_asc",
    "高评分": "rating_desc",
    "评分高": "rating_desc",
    "口碑好": "popular_desc",
    "销量高": "popular_desc",
    "热门": "popular_desc",
    "性价比": "value_desc",
}


def category_label(category: str) -> str:
    return CATEGORY_LABELS.get(category, category)


def category_tags(category: str) -> list[str]:
    return [category, category_label(category), *CATEGORY_SEARCH_TERMS.get(category, [])]
