from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a runtime dependency
    load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parents[1]
if load_dotenv is not None:
    load_dotenv(ROOT_DIR / ".env")


@dataclass(slots=True)
class Settings:
    root_dir: Path
    database_path: Path
    checkpoint_path: Path
    eval_holdout_path: Path
    artifacts_dir: Path
    web_dir: Path
    vector_dim: int
    embedding_model: str
    embedding_index_dir: Path
    embedding_device: str
    embedding_min_score: float
    commerce_platform: str
    policy_vector_backend: str
    policy_qdrant_url: str
    policy_qdrant_api_key: str
    policy_qdrant_path: Path
    policy_qdrant_collection: str
    policy_embedding_model: str
    policy_reranker_model: str
    policy_reranker_enabled: bool
    policy_rrf_k: int
    policy_dense_limit: int
    policy_bm25_limit: int
    policy_fusion_limit: int
    policy_followup_enabled: bool
    policy_max_extra_hops: int
    policy_followup_timeout_seconds: float
    policy_max_evidence_chunks: int
    policy_evidence_token_budget: int
    policy_followup_llm_token_budget: int
    policy_faithfulness_timeout_seconds: float
    policy_faithfulness_llm_token_budget: int
    candidate_pool_limit: int
    eval_sample_rate: float
    llm_mode: str
    intent_mode: str
    api_key: str
    allow_open_api: bool
    api_user_id: str
    auth_users_json: str
    auth_cookie_secure: bool
    auth_session_seconds: int


_HAS_LLM_CREDENTIALS = bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))


def _env(name: str, default: str = "") -> str:
    """Prefer the current prefix while accepting existing deployment settings."""
    value = os.getenv(name)
    if value is not None:
        return value
    if name.startswith("RETAILPILOT_"):
        return os.getenv(name.replace("RETAILPILOT_", "ECOMMERCE_", 1), default)
    return default


def _default_database_path() -> Path:
    database = ROOT_DIR / "data" / "retailpilot.db"
    legacy_database = ROOT_DIR / "data" / "ecommerce.db"
    if not database.exists() and legacy_database.is_file():
        return legacy_database
    return database


def _project_path(name: str, default: Path) -> Path:
    path = Path(_env(name, str(default)))
    return path if path.is_absolute() else ROOT_DIR / path


settings = Settings(
    root_dir=ROOT_DIR,
    database_path=_project_path("RETAILPILOT_DB_PATH", _default_database_path()),
    checkpoint_path=_project_path(
        "RETAILPILOT_CHECKPOINT_PATH", ROOT_DIR / "data" / "checkpoints.db"
    ),
    eval_holdout_path=_project_path(
        "RETAILPILOT_EVAL_HOLDOUT_PATH", ROOT_DIR / "data" / "eval_holdout.db"
    ),
    artifacts_dir=_project_path("RETAILPILOT_ARTIFACTS_DIR", ROOT_DIR / "artifacts"),
    web_dir=_project_path("RETAILPILOT_WEB_DIR", ROOT_DIR / "web"),
    vector_dim=int(_env("RETAILPILOT_VECTOR_DIM", "384")),
    # Neural indexes are promoted explicitly after shadow-eval gates. Keeping
    # hash as the default prevents a newly built local index from silently
    # changing production ranking before it has passed regression.
    embedding_model=_env("RETAILPILOT_EMBEDDING_MODEL", "hash").strip(),
    embedding_index_dir=_project_path(
        "RETAILPILOT_EMBEDDING_INDEX_DIR", ROOT_DIR / "data" / "embeddings"
    ),
    embedding_device=_env("RETAILPILOT_EMBEDDING_DEVICE", "auto").strip().lower(),
    embedding_min_score=float(_env("RETAILPILOT_EMBEDDING_MIN_SCORE", "0.42")),
    commerce_platform=_env("RETAILPILOT_PLATFORM", "demo-mall").strip() or "demo-mall",
    # Policy retrieval never fabricates semantic similarity with hash vectors.
    # `disabled` keeps the base install lightweight; install the policy-RAG
    # extras, build the Qdrant collection, then set this to `qdrant`.
    policy_vector_backend=os.getenv("POLICY_VECTOR_BACKEND", "disabled")
    .strip()
    .lower(),
    policy_qdrant_url=os.getenv("POLICY_QDRANT_URL", "").strip(),
    policy_qdrant_api_key=os.getenv("POLICY_QDRANT_API_KEY", "").strip(),
    policy_qdrant_path=_project_path(
        "POLICY_QDRANT_PATH", ROOT_DIR / "data" / "qdrant"
    ),
    policy_qdrant_collection=os.getenv(
        "POLICY_QDRANT_COLLECTION", "ecommerce_policy_chunks"
    ).strip(),
    policy_embedding_model=os.getenv(
        "POLICY_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
    ).strip(),
    policy_reranker_model=os.getenv(
        "POLICY_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
    ).strip(),
    policy_reranker_enabled=os.getenv("POLICY_RERANKER_ENABLED", "false")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"},
    policy_rrf_k=max(1, int(os.getenv("POLICY_RRF_K", "60"))),
    policy_dense_limit=max(1, int(os.getenv("POLICY_DENSE_LIMIT", "40"))),
    policy_bm25_limit=max(1, int(os.getenv("POLICY_BM25_LIMIT", "40"))),
    policy_fusion_limit=max(1, int(os.getenv("POLICY_FUSION_LIMIT", "30"))),
    policy_followup_enabled=os.getenv("POLICY_FOLLOWUP_ENABLED", "true").lower()
    in {"1", "true", "yes", "on"},
    policy_max_extra_hops=max(0, min(4, int(os.getenv("POLICY_MAX_EXTRA_HOPS", "2")))),
    policy_followup_timeout_seconds=max(
        0.1, float(os.getenv("POLICY_FOLLOWUP_TIMEOUT_SECONDS", "45"))
    ),
    policy_max_evidence_chunks=max(
        1, min(24, int(os.getenv("POLICY_MAX_EVIDENCE_CHUNKS", "12")))
    ),
    policy_evidence_token_budget=max(
        1, int(os.getenv("POLICY_EVIDENCE_TOKEN_BUDGET", "10000"))
    ),
    policy_followup_llm_token_budget=max(
        1, int(os.getenv("POLICY_FOLLOWUP_LLM_TOKEN_BUDGET", "80000"))
    ),
    policy_faithfulness_timeout_seconds=max(
        0.1, float(os.getenv("POLICY_FAITHFULNESS_TIMEOUT_SECONDS", "30"))
    ),
    policy_faithfulness_llm_token_budget=max(
        1, int(os.getenv("POLICY_FAITHFULNESS_LLM_TOKEN_BUDGET", "80000"))
    ),
    candidate_pool_limit=int(_env("RETAILPILOT_CANDIDATE_POOL_LIMIT", "0")),
    eval_sample_rate=max(
        0.0, min(1.0, float(_env("RETAILPILOT_EVAL_SAMPLE_RATE", "0.05")))
    ),
    # With credentials present, LLM understanding is the primary path. Both
    # stages still have deterministic fallbacks for local/offline operation.
    llm_mode=os.getenv("LLM_MODE", "fallback" if _HAS_LLM_CREDENTIALS else "off")
    .strip()
    .lower(),
    intent_mode=os.getenv("INTENT_MODE", "llm" if _HAS_LLM_CREDENTIALS else "rules")
    .strip()
    .lower(),
    api_key=_env("RETAILPILOT_API_KEY", "").strip(),
    allow_open_api=_env("RETAILPILOT_ALLOW_OPEN_API", "").strip().lower()
    in {"1", "true", "yes", "on"},
    api_user_id=_env("RETAILPILOT_API_USER_ID", "service-user"),
    auth_users_json=_env("RETAILPILOT_AUTH_USERS_JSON", ""),
    auth_cookie_secure=_env("RETAILPILOT_AUTH_COOKIE_SECURE", "true").lower()
    in {"1", "true", "yes", "on"},
    auth_session_seconds=max(
        60, min(86400, int(_env("RETAILPILOT_AUTH_SESSION_SECONDS", "3600")))
    ),
)


def ensure_directories() -> None:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.embedding_index_dir.mkdir(parents=True, exist_ok=True)
    settings.policy_qdrant_path.parent.mkdir(parents=True, exist_ok=True)
