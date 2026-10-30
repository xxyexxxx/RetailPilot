from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from .database import db_session, json_dumps, json_loads, utc_now
from .llm_config import invoke_judge
from .utils import _parse_json


RUBRIC_VERSION = "search-relevance-v2"


class JudgedItem(BaseModel):
    sku: str
    relevance: int = Field(ge=0, le=3)
    reason: str = Field(max_length=300)


class SearchJudgeVerdict(BaseModel):
    overall_score: int = Field(ge=0, le=3)
    task_coverage: int = Field(ge=0, le=3)
    ranking_quality: int = Field(ge=0, le=3)
    failure_type: Literal[
        "none",
        "query_understanding",
        "over_filtering",
        "under_filtering",
        "recall",
        "ranking",
        "result_diversity",
        "insufficient_evidence",
    ]
    summary: str = Field(max_length=600)
    items: list[JudgedItem] = Field(default_factory=list, max_length=10)


def _cache_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        {"rubric": RUBRIC_VERSION, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _prompt(payload: dict[str, Any]) -> str:
    return f"""
你是电商搜索离线评审。只评估语义相关性、任务覆盖和排序质量；不要评判价格/品牌/库存等
已由程序校验的硬约束，也不要因为商品文案风格给高分。

评分：0=完全错误，1=弱相关/大部分需求缺失，2=基本相关但有明显遗漏，3=高度相关且排序合理。
failure_type 必须从 none/query_understanding/over_filtering/under_filtering/recall/ranking/
result_diversity/insufficient_evidence 中选择。证据不足时选择 insufficient_evidence，不要猜测。

返回严格 JSON：
{{
  "overall_score": 0,
  "task_coverage": 0,
  "ranking_quality": 0,
  "failure_type": "none",
  "summary": "简短依据",
  "items": [{{"sku": "...", "relevance": 0, "reason": "..."}}]
}}

待评样本：
{json.dumps(payload, ensure_ascii=False)}
""".strip()


def judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    key = _cache_key(payload)
    with db_session() as connection:
        cached = connection.execute(
            "SELECT verdict_json FROM llm_judge_cache WHERE cache_key = ?", (key,)
        ).fetchone()
    if cached:
        return json_loads(cached["verdict_json"], {})

    verdict = SearchJudgeVerdict.model_validate(
        _parse_json(invoke_judge(_prompt(payload)))
    ).model_dump()
    verdict["judge_mode"] = "llm"
    with db_session() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO llm_judge_cache(
                cache_key, rubric_version, verdict_json
            ) VALUES(?, ?, ?)
            """,
            (key, RUBRIC_VERSION, json_dumps(verdict)),
        )
    return verdict


def judge_pending_samples(limit: int = 20) -> dict[str, int]:
    with db_session() as connection:
        rows = connection.execute(
            """
            SELECT sample_id, payload_json
            FROM evaluation_samples
            WHERE status = 'pending'
            ORDER BY created_at
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    completed = 0
    failed = 0
    for row in rows:
        payload = json_loads(row["payload_json"], {})
        try:
            verdict = judge_payload(payload)
        except Exception as exc:
            failed += 1
            with db_session() as connection:
                connection.execute(
                    """
                    UPDATE evaluation_samples
                    SET status = 'judge_error', judge_json = ?, updated_at = ?
                    WHERE sample_id = ?
                    """,
                    (json_dumps({"error": str(exc)[:500]}), utc_now(), row["sample_id"]),
                )
            continue
        completed += 1
        with db_session() as connection:
            connection.execute(
                """
                UPDATE evaluation_samples
                SET status = 'judged', judge_json = ?, updated_at = ?
                WHERE sample_id = ?
                """,
                (json_dumps(verdict), utc_now(), row["sample_id"]),
            )
    return {"selected": len(rows), "completed": completed, "failed": failed}
