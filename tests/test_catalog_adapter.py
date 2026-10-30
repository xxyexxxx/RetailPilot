from RetailPilot.catalog_adapters import normalize_shopsimulator_item


def test_shopsimulator_adapter_preserves_domestic_commerce_fields():
    product = normalize_shopsimulator_item(
        {
            "asin": "123",
            "domain_zh": "家居家装",
            "title": "天然乳胶枕",
            "shop_name": "测试旗舰店",
            "category": "床品›枕头",
            "pricing": [299],
            "attribute": ["天然", "护颈"],
            "images": [],
            "customization_options": {
                "颜色": [
                    {
                        "value": "白色",
                        "price": 299,
                        "is_available": True,
                        "asin": "12301",
                    }
                ]
            },
        }
    )
    assert product["sku"] == "tb-123"
    assert product["commerce"]["currency"] == "CNY"
    assert product["commerce"]["variants"]["颜色"][0]["value"] == "白色"

