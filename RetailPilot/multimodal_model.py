"""One runtime generative transport; embeddings, reranker and Judge are independent."""

from __future__ import annotations

import base64
import io
import json
import os
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from hashlib import sha256
from types import SimpleNamespace
from typing import Literal, TypedDict

import requests
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .execution import BoundedPool, call_deadline

capability_probe = ContextVar("reasoning_capability_probe", default=False)


def endpoint_fingerprint(c):
    return sha256(c.base_url.encode()).hexdigest()


online_pool = BoundedPool("reasoning-online", 4, 8)
offline_pool = BoundedPool("reasoning-offline", 1, 2)


class ModelError(RuntimeError):
    """Safe error code; provider bodies and credentials are never exposed."""


class ImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_kind: Literal["evidence", "asset_bytes", "probe"] = "evidence"
    unit_id: str | None = None
    asset_id: str | None = None
    version: str | None = None
    data: bytes = Field(repr=False)
    mime_type: Literal["image/png", "image/jpeg", "image/webp"] = "image/png"

    @model_validator(mode="after")
    def identity(self):
        if self.source_kind == "evidence" and not (
            self.unit_id and self.asset_id and self.version
        ):
            raise ValueError("bound image input requires source identity")
        return self


@dataclass(frozen=True)
class ModelConfig:
    model: str
    base_url: str
    api_key: str = field(repr=False)
    vision: bool = False
    structured: bool = False
    enabled: bool = False


def config(task_type="reply"):
    # An explicit unified setting wins even when empty; legacy vision selects
    # the shared service for ALL tasks, rather than creating a second client.
    def pick(new, old, default=""):
        return os.environ[new] if new in os.environ else os.getenv(old, default)

    from .asset_config import asset_settings as legacy
    from .evidence_runtime import mode as evidence_mode

    legacy_mode = evidence_mode() in {"legacy", "shadow"}
    vision_task = task_type in {"visual_verification", "image_description"}
    use_old_vision = legacy_mode and vision_task
    old_model = (
        legacy.vlm_model if use_old_vision else os.getenv("LLM_MODEL", "gpt-4o-mini")
    )
    old_base = (
        legacy.vlm_url
        if use_old_vision
        else os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    )
    old_key = (
        legacy.vlm_key
        if use_old_vision
        else os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    )
    if not legacy_mode:
        old_model = legacy.vlm_model or old_model
        old_base = legacy.vlm_url or old_base
        old_key = legacy.vlm_key or old_key
    model = os.getenv("RETAILPILOT_REASONING_MODEL", old_model)
    base = os.getenv("RETAILPILOT_REASONING_BASE_URL", old_base)
    key = os.getenv("RETAILPILOT_REASONING_API_KEY", old_key)
    suffix = "".join(c if c.isalnum() else "_" for c in task_type).upper()
    model = os.getenv("RETAILPILOT_REASONING_MODEL_" + suffix, model)
    enabled = os.getenv("RETAILPILOT_REASONING_ENABLED", "true").lower() not in {
        "0",
        "false",
        "off",
    }
    return ModelConfig(
        model,
        base,
        key,
        os.getenv(
            "RETAILPILOT_REASONING_SUPPORTS_VISION",
            "true" if use_old_vision and legacy.vlm_model else "false",
        ).lower()
        == "true",
        os.getenv("RETAILPILOT_REASONING_SUPPORTS_STRUCTURED_OUTPUT", "false").lower()
        == "true",
        enabled and bool(model and base) and (bool(key) or use_old_vision),
    )


def ready_for_vision(task_type="visual_verification"):
    c = config(task_type)
    if not c.enabled or not c.vision:
        return False
    from .evidence_runtime import mode as evidence_mode

    if evidence_mode() in {"legacy", "shadow"} or capability_probe.get():
        return True
    from pathlib import Path

    try:
        report = json.loads(
            Path(os.getenv("RETAILPILOT_REASONING_CAPABILITIES_FILE", "")).read_text(
                encoding="utf-8"
            )
        )
        return (
            report.get("model") == c.model
            and report.get("endpoint_fingerprint") == endpoint_fingerprint(c)
            and report.get("vision_verified") is True
        )
    except (ValueError, OSError):
        return False


def _content(messages):
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    result = []
    for item in messages:
        if isinstance(item, dict):
            result.append(dict(item))
        else:
            role = {"human": "user", "ai": "assistant"}.get(
                getattr(item, "type", ""), getattr(item, "type", "user")
            )
            result.append({"role": role, "content": item.content})
    return result


def strict_schema(model):
    """Normalize the closed output protocol for strict JSON Schema providers."""
    schema = model.model_json_schema()

    def visit(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    return schema


class GatewayModel:
    def __init__(
        self,
        *,
        task_type="reply",
        temperature=0,
        json_mode=False,
        max_tokens=None,
        schema=None,
        timeout=None,
        image_count=0,
    ):
        self.task_type, self.temperature, self.json_mode = (
            task_type,
            temperature,
            json_mode,
        )
        self.max_tokens, self.schema, self.timeout = max_tokens, schema, timeout
        self.image_count = image_count

    def invoke(self, messages):
        from .llm_config import agent_max_tokens, policy_call_budget

        c = config(self.task_type)
        if not c.enabled:
            raise ModelError("model_unavailable")
        if self.image_count and not c.vision:
            raise ModelError("unsupported_vision")
        timeout = self.timeout or float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
        deadlines = [
            x
            for x in (
                call_deadline.get(),
                getattr(policy_call_budget.get(), "deadline", None),
            )
            if x is not None
        ]
        if deadlines:
            timeout = min(timeout, min(deadlines) - time.perf_counter())
        if timeout <= 0:
            raise TimeoutError("model_deadline")
        payload = {
            "model": c.model,
            "messages": _content(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens or agent_max_tokens(self.task_type),
            "stream": False,
        }
        if self.schema is not None and c.structured:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": self.schema.__name__,
                    "strict": True,
                    "schema": strict_schema(self.schema),
                },
            }
        elif self.json_mode or self.schema:
            payload["response_format"] = {"type": "json_object"}
        from .evidence_runtime import mode as evidence_mode

        old_thinking = (
            evidence_mode() in {"legacy", "shadow"}
            and "RETAILPILOT_REASONING_MODEL" not in os.environ
            and "LLM_THINKING" in os.environ
        )
        if (
            old_thinking
            or os.getenv("RETAILPILOT_REASONING_SEND_THINKING", "false") == "true"
        ):
            payload["thinking"] = {
                "type": "enabled"
                if os.getenv("LLM_THINKING", "false") == "true"
                else "disabled"
            }
        url = c.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        try:
            # Single attempt. Callers own the end-to-end retry and time budget.
            with requests.post(
                url,
                json=payload,
                headers={"Authorization": "Bearer " + c.api_key},
                timeout=timeout,
                allow_redirects=False,
            ) as response:
                if response.status_code >= 300:
                    raise ModelError(
                        {
                            429: "rate_limit",
                            400: "invalid_request",
                            413: "context_overflow",
                        }.get(response.status_code, "provider_error")
                    )
                data = response.json()
        except requests.Timeout as exc:
            raise TimeoutError("provider_timeout") from exc
        except requests.RequestException as exc:
            raise ModelError("provider_connection_error") from exc
        try:
            choice = data["choices"][0]
            message = choice["message"]
            if message.get("refusal"):
                raise ModelError("model_refusal")
            if choice.get("finish_reason") not in (None, "stop"):
                raise ModelError("truncated_output")
            text = message["content"]
            if not isinstance(text, str) or not text.strip():
                raise ModelError("invalid_output")
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelError("invalid_output") from exc
        return SimpleNamespace(
            content=text, response_metadata={"token_usage": data.get("usage", {})}
        )


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict]


class MultimodalModelGateway:
    def generate(
        self,
        *,
        task_type: str,
        messages: list[ChatMessage] | str,
        images: list[ImageInput] | None = None,
        output_schema: type[BaseModel] | None = None,
        timeout_seconds: float | None = None,
        lane: Literal["online", "offline"] = "online",
        max_tokens: int | None = None,
    ):
        from .llm_config import invoke_with_telemetry

        images = images or []
        if len(images) > int(os.getenv("RETAILPILOT_EVIDENCE_MAX_IMAGES", "2")):
            raise ModelError("image_budget")
        wire = _content(messages)
        if images:
            from PIL import Image

            if not ready_for_vision(task_type):
                raise ModelError("unsupported_vision")
            parts = []
            for item in images:
                try:
                    with Image.open(io.BytesIO(item.data)) as img:
                        if img.width * img.height > 20_000_000:
                            raise ModelError("invalid_image")
                        img.verify()
                except (OSError, ValueError) as exc:
                    raise ModelError("invalid_image") from exc
                if len(item.data) > 16 * 1024 * 1024:
                    raise ModelError("invalid_image")
                parts.extend(
                    [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "unit_id": item.unit_id,
                                    "asset_id": item.asset_id,
                                    "version": item.version,
                                }
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{item.mime_type};base64,"
                                + base64.b64encode(item.data).decode()
                            },
                        },
                    ]
                )
            wire.append({"role": "user", "content": parts})
        model = GatewayModel(
            task_type=task_type,
            json_mode=True,
            schema=output_schema,
            timeout=timeout_seconds,
            image_count=len(images),
            max_tokens=max_tokens,
        )
        timeout = timeout_seconds or float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
        deadline = min(
            call_deadline.get() or float("inf"), time.perf_counter() + timeout
        )
        response = (offline_pool if lane == "offline" else online_pool).call(
            lambda: invoke_with_telemetry(model, wire, prompt_kind=task_type), deadline
        )
        result = json.loads(response.content)
        if output_schema is None and not isinstance(result, dict):
            raise ModelError("invalid_json_object")
        return output_schema.model_validate(result) if output_schema else result


gateway = MultimodalModelGateway()
