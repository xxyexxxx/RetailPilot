from __future__ import annotations

from datetime import date

import RetailPilot.database as database_module
from RetailPilot.config import settings
from RetailPilot.policy import _rrf, retrieve_policies
from RetailPilot.policy_ingestion import (
    PolicyDocumentInput,
    chunk_policy_document,
    ingest_policy_documents,
)


def _document(
    doc_id: str,
    *,
    topic: str = "退款",
    content: str,
    platform: str = "policy-test",
    audience: str = "customer",
    effective_from: str = "2026-01-01",
    effective_to: str = "",
) -> PolicyDocumentInput:
    return PolicyDocumentInput(
        doc_id=doc_id,
        policy_id=doc_id,
        topic=topic,
        title="测试政策",
        content=content,
        source_name="test-fixture",
        source_uri=f"memory://{doc_id}",
        platform=platform,
        audience=audience,
        effective_from=effective_from,
        effective_to=effective_to,
    )


def test_short_markdown_sections_are_independent_leaf_chunks() -> None:
    document = _document(
        "policy-short-sections",
        content="# 退款政策\n\n## 信用卡\n审核后原路退回。\n\n## 余额\n审核后退回账户余额。",
    )

    chunks = chunk_policy_document(document)

    assert [item["chunk_type"] for item in chunks] == ["leaf", "leaf"]
    assert all(item["parent_chunk_id"] is None for item in chunks)
    assert chunks[0]["section_path"].endswith("信用卡")


def test_long_section_indexes_children_and_bounds_parent_size() -> None:
    body = "".join(
        f"第{index}条：退款审核完成后按原支付路径处理，具体到账时间以支付机构为准。"
        for index in range(180)
    )
    chunks = chunk_policy_document(
        _document("policy-parent-child", content=f"# 退款政策\n\n## 到账规则\n{body}")
    )
    parents = [item for item in chunks if item["chunk_type"] == "parent"]
    children = [item for item in chunks if item["chunk_type"] == "child"]

    assert parents
    assert children
    assert all(item["token_count"] <= 1200 for item in parents)
    assert all(item["token_count"] <= 350 for item in children)
    assert {item["parent_chunk_id"] for item in children} <= {
        item["chunk_id"] for item in parents
    }


def test_policy_retrieval_hard_filters_before_bm25(monkeypatch) -> None:
    ingest_policy_documents(
        [
            _document(
                "policy-public-refund",
                content="# 退款\n\n## 信用卡到账\n信用卡退款通常在三到十个工作日到账。",
            ),
            _document(
                "policy-wrong-topic",
                topic="发票",
                content="# 发票\n\n## 开票\n订单完成后可以申请电子发票。",
            ),
            _document(
                "policy-expired-refund",
                content="# 旧退款\n\n## 到账\n旧规则为一个工作日到账。",
                effective_to="2026-06-01",
            ),
            _document(
                "policy-internal-refund",
                content="# 内部退款\n\n## 审核\n内部审核流程不得对外展示。",
                audience="internal",
            ),
        ],
        database_path=settings.database_path,
        source_version="test-v1",
    )
    monkeypatch.setattr(settings, "policy_vector_backend", "disabled")
    monkeypatch.setattr(settings, "policy_reranker_enabled", False)
    monkeypatch.setattr(
        database_module,
        "refresh_policy_fts",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("policy retrieval must not rebuild FTS")
        ),
    )

    result = retrieve_policies(
        "信用卡退款多久到账",
        topic="退款",
        platform="policy-test",
        as_of=date(2026, 8, 23),
    )

    assert [item["doc_id"] for item in result["documents"]] == [
        "policy-public-refund"
    ]
    assert result["trace"]["hard_filter"]["candidate_chunks"] == 1
    assert result["trace"]["dense"] | {
        "available": False,
        "error": "disabled",
    } == result["trace"]["dense"]
    assert "hash" not in str(result["trace"]["dense"]).lower()
    citation = result["documents"][0]["citation"]
    assert citation["chunk_id"]
    assert citation["quoted_text"] == result["documents"][0]["content"]


def test_rrf_fuses_rankings_without_raw_score_addition() -> None:
    ranking, scores, source_ranks = _rrf(
        [["a", "b"], ["b", "c"]], k=60, limit=3
    )

    assert ranking[0] == "b"
    assert source_ranks["b"] == [2, 1]
    assert scores["b"] > scores["a"]


def test_followup_preserves_scope_date_versions_and_stale_dense_fence(monkeypatch):
    """Exercise real SQL/FTS on BOTH topics, including stale vector candidates."""
    import RetailPilot.policy as policy_module
    from RetailPilot.policy_followup import run_policy_followup
    from RetailPilot.policy_index import PolicyDenseRecall

    valid = _document("adaptive-valid", topic="保修", content="质量问题超过七天可以申请核验。", platform="adaptive-test")
    invalid = [
        _document("adaptive-old", topic="保修", content="质量问题无条件保证退货。", platform="adaptive-test", effective_to="2026-01-02"),
        _document("adaptive-other-platform", topic="保修", content="质量问题无条件保证退货。", platform="wrong"),
        _document("adaptive-internal", topic="保修", content="质量问题内部规则。", platform="adaptive-test", audience="internal"),
        _document("adaptive-category", topic="保修", content="质量问题其他类目。", platform="adaptive-test"),
        _document("adaptive-inactive", topic="保修", content="质量问题失效版本。", platform="adaptive-test"),
        _document("adaptive-future", topic="保修", content="质量问题未来规则。", platform="adaptive-test", effective_from="2027-01-01"),
    ]
    invalid[3].category_scope = "不适用类目"
    invalid[4].status = "superseded"
    invalid[4].version = "old"
    ingest_policy_documents([valid, *invalid], database_path=settings.database_path, source_version="adaptive-filter-test")
    from RetailPilot.policy_ingestion import chunk_policy_document
    stale_ids = [item["chunk_id"] for document in invalid for item in chunk_policy_document(document)]
    class StaleDense:
        def search(self, *a, **kw):
            return PolicyDenseRecall(ranking=stale_ids, scores={cid: 1 for cid in stale_ids},
                                     backend="fake", model="fake", index_version="stale", available=True)
    monkeypatch.setattr(policy_module, "get_policy_dense_index", lambda: StaleDense())
    initial = retrieve_policies("退货质量问题", topic="退换货", platform="adaptive-test", as_of=date(2026, 8, 23))
    assert not initial["documents"]
    seen = []
    def assess(query, *, documents, **kw):
        if not documents:
            return {"decision": "clarify", "next_action": "retrieve_more", "sufficient": False, "conflict": False,
                    "reason": "缺少质量例外", "missing_aspects": ["质量例外"],
                    "suggested_queries": [{"aspect": "质量例外", "query": "质量例外", "topic": "保修"}]}
        seen.extend(documents)
        ids = [item["chunk_id"] for item in documents]
        return {"decision": "answer", "next_action": "answer", "sufficient": True, "conflict": False,
                "reason": "固定夹具已覆盖", "covered_aspects": ["质量例外"], "supporting_chunk_ids": ids,
                "aspect_evidence_map": {"质量例外": ids}}
    def retrieve(args):
        return {"ok": True, "data": retrieve_policies(**args)}
    result = run_policy_followup(query="退货质量问题", topic="退换货", initial_payload=initial,
        assess=assess, retrieve=retrieve,
        retrieval_arguments={"platform": "adaptive-test", "as_of": date(2026, 8, 23)})
    assert result["summary"]["stop_reason"] == "answer"
    assert {item["doc_id"] for item in seen} == {"adaptive-valid"}
    assert not (set(result["summary"]["selected_chunk_ids"]) & set(stale_ids))
