from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RetailPilot.catalog_adapters import normalize_shopsimulator_item, upsert_normalized_products
from RetailPilot.config import settings
from RetailPilot.database import db_session, init_db, json_dumps, set_metadata
from RetailPilot.eval_dataset import seed_retrieval_eval_dataset
from RetailPilot.text import hash_vector, vector_to_blob


CATEGORY_PROFILES = {
    "Computers": [
        ("Windows Office Laptop", "laptop notebook windows business office productivity"),
        ("Intel Business Laptop", "laptop notebook intel business office"),
        ("Touchscreen Laptop", "laptop notebook touchscreen touch screen"),
        ("Gaming Laptop", "laptop notebook gaming game"),
        ("Chromebook Laptop", "laptop notebook chromebook chrome os"),
        ("Lightweight Travel Laptop", "laptop notebook lightweight thin portable travel"),
    ],
    "Cell Phones & Accessories": [
        ("Fast Phone Charger", "phone accessory charger charging adapter fast quick charge usb c"),
        ("Portable Power Bank", "phone accessory power bank portable charger"),
        ("Durable Phone Cable", "phone accessory cable cord"),
        ("Tempered Screen Protector", "phone accessory screen protector tempered glass"),
        ("Car Phone Charger", "phone accessory car charger vehicle charger car charging"),
        ("Wireless Charging Pad", "phone accessory wireless charging charging pad"),
    ],
    "All Beauty": [
        ("Gentle Moisturizing Cream", "beauty sensitive gentle hypoallergenic moisturizer moisturizing hydrating cream"),
        ("Light Serum Essence", "beauty serum essence"),
        ("Sensitive Skin Care", "beauty sensitive gentle hypoallergenic skincare"),
        ("Fresh Perfume", "beauty perfume fragrance eau de fresh"),
        ("Makeup Brush", "beauty brush applicator makeup"),
        ("Lip Balm", "beauty lip balm moisturizing"),
    ],
    "Sports & Outdoors": [
        ("Waterproof Sports Gear", "sports outdoor waterproof water resistant"),
        ("Yoga Stretch Mat", "sports yoga stretch fitness"),
        ("Camping Tent", "sports outdoor camping camp tent"),
        ("Running Hydration Bottle", "sports running bottle hydration"),
        ("Cycling Gear", "sports bike bicycle cycling"),
        ("Ultralight Hiking Gear", "sports outdoor lightweight ultralight light weight hiking"),
    ],
    "Appliances": [
        ("Coffee Appliance", "appliance home coffee espresso"),
        ("Floor Vacuum Cleaner", "appliance home vacuum floor cleaner clean"),
        ("Room Heater", "appliance home heater heating"),
        ("Air Circulator Fan", "appliance home fan air circulator"),
        ("Kitchen Cleaning Appliance", "appliance home kitchen clean wash"),
    ],
    "Video Games": [
        ("Nintendo Switch Controller", "video game gaming nintendo switch gamepad controller"),
        ("PlayStation Controller", "video game gaming playstation ps4 ps5 controller"),
        ("Xbox Controller", "video game gaming xbox controller gamepad"),
        ("Gaming Voice Headset", "video game gaming headset headphone microphone"),
        ("Wireless Gamepad", "video game gaming gamepad controller wireless bluetooth"),
    ],
}


def _legacy_products(per_category: int = 600):
    categories = ["Electronics", *CATEGORY_PROFILES]
    for category in categories:
        for index in range(per_category):
            if category == "Electronics":
                if index < 5:
                    title, content = (
                        "Wireless Noise Cancelling Headphone",
                        "electronics headphone headset wireless bluetooth noise cancelling noise canceling ANC",
                    )
                elif index < 100:
                    title, content = (
                        "Wireless Over Ear Headphone",
                        "electronics headphone headset wireless bluetooth over ear",
                    )
                else:
                    profiles = [
                        ("Bluetooth Speaker", "electronics speaker wireless bluetooth"),
                        ("Portable Travel Camera", "electronics camera portable compact travel photography"),
                        ("USB Meeting Microphone", "electronics microphone mic computer usb meeting"),
                        ("Fitness Smart Watch", "electronics smartwatch smart watch fitness health sport"),
                        ("Wireless Gaming Mouse", "electronics mouse mice wireless bluetooth gaming rgb"),
                        ("Mechanical Gaming Keyboard", "electronics keyboard keypad mechanical gaming rgb"),
                    ]
                    profile_index = min(len(profiles) - 1, (index - 100) // 84)
                    title, content = profiles[profile_index]
            else:
                profiles = CATEGORY_PROFILES[category]
                profile_index = min(len(profiles) - 1, index * len(profiles) // per_category)
                title, content = profiles[profile_index]
            price = float(20 + index % 80)
            sku = f"ci-{category.lower().replace(' ', '-').replace('&', 'and')}-{index:03d}"
            searchable = f"{title} {content} {category} item {index}"
            yield {
                "sku": sku,
                "title": f"{title} {index}",
                "brand": "CI Brand",
                "category": category,
                "category_label": category,
                "description": content,
                "price": price,
                "rating": 4.0 + (index % 10) / 10,
                "rating_count": 100 + index,
                "stock": 10,
                "features": content.split()[:20],
                "attributes": {"keywords": content},
                "tags": content.split(),
                "tags_text": content,
                "attributes_text": content,
                "image_url": "",
                "source": "ci-synthetic",
                "source_url": "",
                "quality_score": 0.8,
                "popularity_score": 0.7,
                "vector": vector_to_blob(hash_vector(searchable)),
                "commerce": {
                    "platform": "ci", "platform_product_id": sku, "spu_id": sku,
                    "seller_name": "CI Store", "currency": "USD", "category_path": category,
                    "variants": {}, "services": [], "provenance": {"synthetic": True},
                    "dataset_split": "ci", "license_note": "Deterministic synthetic CI fixture",
                },
                "instructions": [], "user_persona": {},
            }


def _domestic_products():
    samples = [
        ("910000000001", "泰国天然乳胶儿童护颈枕 4-6岁满天星", "家居家装", "梦洁宝贝旗舰店", 899),
        ("910000000002", "国产轻薄游戏本 RTX4060 16G 1TB", "家用电器数码", "极客电脑旗舰店", 5999),
        ("910000000003", "婴儿有机棉可拆洗睡袋 四季款", "母婴儿童", "童梦旗舰店", 299),
    ]
    for asin, title, domain, shop, price in samples:
        yield normalize_shopsimulator_item(
            {
                "asin": asin, "tag": "ci", "domain_zh": domain, "title": title,
                "shop_name": shop, "category": f"{domain}›精选", "full_description": title,
                "attribute": title.split(), "pricing": [price], "images": [],
                "customization_options": {
                    "规格": [
                        {"value": "标准款", "is_available": True, "price": price, "asin": f"{asin}01"}
                    ]
                },
                "instructions": [{"instruction": f"帮我购买{title}", "options": ["标准款"], "attributes": title.split()}],
                "user_persona": {},
            }
        )


def main() -> None:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    if settings.database_path.exists():
        settings.database_path.unlink()
    init_db()
    inserted = upsert_normalized_products([*_legacy_products(), *_domestic_products()])
    # Reuse the production seeders only after the deterministic catalog exists.
    from scripts.bootstrap_data import seed_business_data, seed_eval_cases
    seed_business_data()
    seed_eval_cases()
    eval_report = seed_retrieval_eval_dataset()
    set_metadata("dataset_version", "ci-catalog-v1")
    set_metadata("dataset", {"name": "ci-catalog-v1", "products": inserted, "synthetic": True})
    print({"products": inserted, "evaluation": eval_report, "database": str(settings.database_path)})


if __name__ == "__main__":
    main()
