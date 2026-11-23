"""Configuration for optional multimodal processing and retrieval."""

from dataclasses import dataclass, field
from pathlib import Path

from .config import ROOT_DIR, _env, _project_path


def _flag(name: str) -> bool:
    return _env(f"RETAILPILOT_MM_{name}", "0").casefold() in {"1", "true", "yes"}


def _number(name: str, default: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(_env(f"RETAILPILOT_MM_{name}", str(default)))))


@dataclass(slots=True)
class AssetSettings:
    enabled: bool = _flag("ENABLED")
    visual_enabled: bool = _flag("VISUAL_ENABLED")
    store: Path = field(
        default_factory=lambda: _project_path(
            "RETAILPILOT_MM_STORE", ROOT_DIR / "data" / "assets"
        )
    )
    curators: str = _env("RETAILPILOT_MM_CURATORS")
    remote_hosts: str = _env("RETAILPILOT_MM_REMOTE_HOSTS")
    parser: str = _env("RETAILPILOT_MM_PARSER", "docling")
    docling_artifacts: str = _env("RETAILPILOT_MM_DOCLING_ARTIFACTS")
    ocr: str = _env("RETAILPILOT_MM_OCR", "off")
    ocr_languages: str = _env("RETAILPILOT_MM_OCR_LANGUAGES", "chi_sim+eng")
    vlm_url: str = _env("RETAILPILOT_MM_VLM_URL")
    vlm_key: str = _env("RETAILPILOT_MM_VLM_KEY")
    vlm_model: str = _env("RETAILPILOT_MM_VLM_MODEL")
    visual_model: str = _env(
        "RETAILPILOT_MM_VISUAL_MODEL", "google/siglip2-base-patch16-224"
    )
    visual_revision: str = _env("RETAILPILOT_MM_VISUAL_REVISION", "main")
    visual_min_score: float = float(_env("RETAILPILOT_MM_VISUAL_MIN_SCORE", "0.12"))
    text_model: str = _env("RETAILPILOT_MM_TEXT_MODEL", "hash")
    text_revision: str = _env("RETAILPILOT_MM_TEXT_REVISION", "main")
    model_local_only: bool = _flag("LOCAL_MODELS_ONLY")
    max_bytes: int = _number("MAX_BYTES", 16 * 1024 * 1024, 1024, 64 * 1024 * 1024)
    max_pages: int = _number("MAX_PAGES", 40, 1, 100)
    max_pixels: int = _number("MAX_PIXELS", 20_000_000, 100_000, 50_000_000)
    tile_height: int = _number("TILE_HEIGHT", 1400, 400, 2400)
    context_tokens: int = _number("CONTEXT_TOKENS", 96, 0, 256)
    top_k: int = _number("TOP_K", 32, 1, 100)
    evidence_per_sku: int = _number("EVIDENCE_PER_SKU", 3, 1, 8)
    verification_images: int = _number("VERIFY_IMAGES", 2, 0, 5)
    verification_tokens: int = _number("VERIFY_TOKENS", 360, 64, 1000)
    verification_timeout: int = _number("VERIFY_TIMEOUT", 8, 1, 30)
    processing_timeout: int = _number("PROCESS_TIMEOUT", 180, 10, 900)
    qdrant_url: str = _env("RETAILPILOT_MM_QDRANT_URL")
    qdrant_key: str = _env("RETAILPILOT_MM_QDRANT_KEY")
    qdrant_collection: str = _env(
        "RETAILPILOT_MM_QDRANT_COLLECTION", "retailpilot_asset_visual_v1"
    )


asset_settings = AssetSettings()
