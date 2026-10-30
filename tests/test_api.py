from fastapi.testclient import TestClient

from RetailPilot.api import app


def test_api_smoke():
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        search = client.post(
            "/api/search",
            json={
                "query": "推荐 50 美元以内的无线耳机",
                "target_terms": ["无线耳机", "耳机"],
                "user_id": "demo-user",
                "limit": 5,
                "mode": "hybrid",
            },
        )
        assert search.status_code == 200
        assert search.json()["products"]

        chat = client.post(
            "/api/chat",
            json={
                "session_id": "api-test",
                "user_id": "demo-user",
                "message": "推荐 50 美元以内的无线耳机",
            },
        )
        assert chat.status_code == 200
        # /api/chat 走 agent 链路：意图由真实 LLM 输出。测试环境无 LLM Key 时
        # 意图返回空任务、回复默认话术（LLM 失效不往后走），不保证有商品。
        assert isinstance(chat.json()["products"], list)
