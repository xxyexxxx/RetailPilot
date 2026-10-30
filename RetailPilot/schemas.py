from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


@dataclass(slots=True)
class ParsedQuery:
    raw_query: str
    rewritten_query: str
    intent: str = "search"
    category: str | None = None
    price_min: float | None = None
    price_max: float | None = None
    brands: list[str] = field(default_factory=list)
    preferred_brands: list[str] = field(default_factory=list)
    negative_brands: list[str] = field(default_factory=list)
    excluded_title_terms: list[str] = field(default_factory=list)
    must_have_groups: list[list[str]] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    grounded_target_terms: list[str] = field(default_factory=list)
    llm_concepts: list[str] = field(default_factory=list)
    concept_evidence: dict[str, str] = field(default_factory=dict)
    negative_terms: list[str] = field(default_factory=list)
    reference_skus: list[str] = field(default_factory=list)
    sort: str = "relevance"
    sources: dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0
    clarification_slot: str | None = None
    clarification_question: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


IntentName = Literal["search", "detail", "compare", "order", "service", "chitchat"]

# 检索类目白名单：ShopSimulator/Taobao 中文类目，LLM 只能从这里选。
# 定义在 schema 层，让 category 的取值在 Pydantic 校验时即被锁定（enum 白名单），
# 白名单外的值直接校验失败、不进入下游，而不是到业务层才被丢弃。
CATEGORY_VALUES = (
    "家居家装",
    "家用电器数码",
    "服饰鞋包饰品",
    "休闲娱乐文教",
    "美妆个护健康",
    "运动户外交通",
    "食品饮品",
    "母婴儿童",
    "生产材料农用品",
)
AllowedCategory = Literal[
    "家居家装",
    "家用电器数码",
    "服饰鞋包饰品",
    "休闲娱乐文教",
    "美妆个护健康",
    "运动户外交通",
    "食品饮品",
    "母婴儿童",
    "生产材料农用品",
]

# 意图分析输出的格式
class IntentTask(BaseModel):
    """One independently executable objective extracted from a user query.

    For `search` intents the LLM also emits the retrieval-relevant signals
    (category + Chinese target terms) in the same call, mirroring how
    agentic product recommendation produces a single structured tool request
    (Gemini Shopping Graph / Algolia Agentic Search). The local query parser
    may still refine these against its own rule layer.
    """

    intent: IntentName
    text: str = Field(min_length=1, max_length=1000)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    category: AllowedCategory | None = Field(default=None)
    # 同时容纳英文标准词 + 中文子品类词（标题是中文，需中文词命中）
    target_terms: list[str] = Field(default_factory=list, max_length=12)
    # Structured retrieval constraints emitted in the same call, mirroring how
    # agentic product recommendation turns one utterance into a single tool
    # request with a constraints object (Gemini Shopping Graph / Algolia).
    price_min: float | None = None
    price_max: float | None = None
    brands: list[str] = Field(default_factory=list, max_length=4)
    negative_brands: list[str] = Field(default_factory=list, max_length=4)
    concepts: list[str] = Field(default_factory=list, max_length=12)
    negative_terms: list[str] = Field(default_factory=list, max_length=6)
    # 信息不足需要追问时，LLM 直接输出澄清（而不是给残缺检索信号）。
    # 由意图层判断"该不该问、问什么"，代码不做 case-by-case 澄清规则。
    clarification_slot: str | None = Field(default=None, max_length=64)
    clarification_question: str | None = Field(default=None, max_length=300)


class IntentAnalysis(BaseModel):
    tasks: list[IntentTask] = Field(min_length=1, max_length=8)


class GroundedQueryTerm(BaseModel):
    """A normalized term whose evidence must occur verbatim in the user query."""

    term: str = Field(min_length=1, max_length=100)
    evidence: str = Field(min_length=1, max_length=200)


class QueryRefinement(BaseModel):
    category: str | None = None
    category_evidence: str | None = Field(default=None, max_length=200)
    target_terms: list[GroundedQueryTerm] = Field(default_factory=list, max_length=4)
    concepts: list[GroundedQueryTerm] = Field(default_factory=list, max_length=6)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    user_id: str = Field(default="demo-user", min_length=1, max_length=80)
    limit: int = Field(default=12, ge=1, le=30)
    mode: Literal["hybrid", "fts"] = "hybrid"
    # 手动调用等价于"人当 LLM 传参"：检索词由调用方给出（FTS 召回唯一来源），
    # query 仅作原文记录/向量召回，不再走本地词表解析。
    target_terms: list[str] = Field(default_factory=list, max_length=12)
    concepts: list[str] = Field(default_factory=list, max_length=12)
    category: str | None = None
    price_min: float | None = Field(default=None, ge=0)
    price_max: float | None = Field(default=None, ge=0)
    brands: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    session_id: str = Field(default="web-demo", min_length=1, max_length=100)
    user_id: str = Field(default="demo-user", min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=1000)


class InteractionRequest(BaseModel):
    user_id: str = Field(default="demo-user", min_length=1, max_length=80)
    sku: str = Field(min_length=2, max_length=40)
    event_type: Literal["impression", "click", "favorite", "cart", "purchase", "dislike"]
    query: str = Field(default="", max_length=500)


class CompareRequest(BaseModel):
    skus: list[str] = Field(min_length=2, max_length=4)
