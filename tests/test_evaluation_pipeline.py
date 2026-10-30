from RetailPilot import retrieval
from RetailPilot.catalog import list_brands
from RetailPilot.config import settings
from RetailPilot.database import db_session, json_loads
from RetailPilot.evaluation import _ndcg, load_qrels
from RetailPilot.query import parse_query
from RetailPilot.retrieval import search_engine


def test_ndcg_uses_full_qrel_ideal_not_only_returned_items():
    score = _ndcg([3.0], [3.0, 3.0, 3.0])
    assert 0 < score < 1


def test_seeded_retrieval_cases_have_graded_qrels():
    qrels = load_qrels()
    assert len(qrels) == 48
    assert sum(len(items) for items in qrels.values()) >= 5000
    assert all(max(items.values()) >= 2 for items in qrels.values())


def test_low_candidate_policy_relaxes_before_zero_results():
    from dataclasses import replace

    base = parse_query(
        "预算100美元以内的无线降噪耳机，排除入耳式，最好索尼",
        known_brands=list_brands(),
    )
    # FTS 召回唯一来源是 LLM 检索词（中文数据 → 中文检索词）；
    # must_have_groups 覆盖为中文属性组（本地解析遗留英文组在中文数据上必空），
    # 一组可命中、一组不存在，验证受控放松先于 0 结果。
    parsed = replace(
        base,
        grounded_target_terms=["无线耳机", "降噪耳机", "耳机"],
        llm_concepts=["降噪"],
        must_have_groups=[["蓝牙耳机"], ["attribute-that-does-not-exist"]],
    )
    result = search_engine.search(parsed, user_id="new-user", limit=10)
    policy = next(item for item in result["trace"] if item["node"] == "candidate_policy")

    # 放松机制核心保证：严格属性候选不足时，受控放松先于 0 结果
    # （补足缺少一组的近似候选）。strict 是否 >0 取决于数据分布。
    assert policy["funnel"]["attribute_policy_pass"] > 0
    assert result["relaxations"]
    assert all(item["price"] <= 100 for item in result["products"])
    assert any(
        item["type"] == "low_candidate_attribute_relaxation"
        for item in result["relaxations"]
    )


def test_search_logging_persists_routes_impressions_and_sample():
    original_rate = settings.eval_sample_rate
    settings.eval_sample_rate = 1.0
    try:
        result = search_engine.search("推荐数码产品", user_id="new-user", limit=5)
    finally:
        settings.eval_sample_rate = original_rate

    with db_session() as connection:
        route_count = connection.execute(
            "SELECT COUNT(*) FROM search_route_logs WHERE request_id = ?",
            (result["request_id"],),
        ).fetchone()[0]
        impressions = connection.execute(
            "SELECT COUNT(*) FROM search_result_impressions WHERE request_id = ?",
            (result["request_id"],),
        ).fetchone()[0]
        sample = connection.execute(
            "SELECT sampling_reasons_json FROM evaluation_samples WHERE request_id = ?",
            (result["request_id"],),
        ).fetchone()

    assert route_count > 0
    assert impressions == 5
    assert "random_sample" in json_loads(sample["sampling_reasons_json"], [])


def test_neural_runtime_failure_falls_back_to_hash(monkeypatch):
    class BrokenIndex:
        def recall(self, *_args, **_kwargs):
            raise OSError("model cache is incomplete")

    monkeypatch.setattr(retrieval, "get_neural_index", lambda: BrokenIndex())
    parsed = parse_query("推荐数码产品")
    pool = retrieval._structured_skus(parsed)
    dense = retrieval._dense_recall_result(parsed, pool, limit=5)

    assert dense.backend == "hash-flat"
    assert dense.ranking
    assert "model cache is incomplete" in dense.fallback_reason
