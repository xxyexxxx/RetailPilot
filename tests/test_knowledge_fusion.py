from RetailPilot.knowledge_retrieval import RetrievalHit, aggregate_skus, fuse


def hit(uid, route, rank, sku="a", content=None):
    return RetrievalHit(
        uid,
        route,
        0.9,
        rank,
        {
            "sku": sku,
            "content_hash": content or uid,
            "search_text": uid,
            "evidence_text": uid,
        },
    )


def test_rrf_votes_once_per_method_and_uses_unit_identity():
    bm25 = hit("profile:a:0", "knowledge_bm25", 1)
    dense = hit("image:a", "knowledge_dense", 1)
    visual = hit("image:a", "knowledge_visual", 1)
    fused = fuse(
        {
            "knowledge_bm25": [bm25, bm25],
            "knowledge_dense": [dense],
            "knowledge_visual": [visual],
        }
    )
    assert [h.unit_id for h in fused] == ["image:a", "profile:a:0"]
    assert fused[0].score == 2 / 61
    assert fused[1].score == 1 / 61
    assert set(fused[0].metadata["source_ranks"]) == {
        "knowledge_dense",
        "knowledge_visual",
    }


def test_many_duplicate_images_cannot_accumulate_unbounded_sku_score():
    few = fuse({"knowledge_bm25": [hit("p", "knowledge_bm25", 1)]})
    many = fuse(
        {
            "knowledge_bm25": [
                hit("p", "knowledge_bm25", 1),
                *[
                    hit(f"img-{i}", "knowledge_bm25", i + 2, content="same-image")
                    for i in range(100)
                ],
                hit("independent", "knowledge_bm25", 103),
            ]
        },
        limit=240,
    )
    base = aggregate_skus(few)["a"]["score"]
    result = aggregate_skus(many, evidence_limit=3)["a"]
    assert base <= result["score"] <= base * 1.1
    assert len(result["hits"]) == 3
    assert len({h.metadata["content_hash"] for h in result["hits"]}) == 3
    # Caption-free images have equal empty text but distinct visual sources.
    left = hit("left", "knowledge_visual", 1, content="empty-text")
    right = hit("right", "knowledge_visual", 2, content="empty-text")
    duplicate = hit("duplicate", "knowledge_visual", 3, content="other-text")
    left.metadata["visual_hash"] = duplicate.metadata["visual_hash"] = "binary-left"
    right.metadata["visual_hash"] = "binary-right"
    visual = aggregate_skus(fuse({"knowledge_visual": [left, right, duplicate]}))["a"]
    assert {h.unit_id for h in visual["hits"]} == {"left", "right"}
