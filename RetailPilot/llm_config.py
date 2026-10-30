"""
模型工厂：统一从这里取 LLM，支持 OpenAI 兼容接口（DeepSeek / 通义 / 智谱 / vLLM 本地等）。

配置方式（优先级从高到低）：
1. 复制 .env.example 为 .env 并填好 LLM_API_KEY
2. 或直接设置环境变量 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
"""

import json
import logging
import os
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from langchain_deepseek import ChatDeepSeek
    _HAS_DEEPSEEK = True
except ImportError:
    _HAS_DEEPSEEK = False

from langchain_openai import ChatOpenAI

from .execution import call_deadline

LOGGER = logging.getLogger(__name__)

class PolicyBudgetExceeded(RuntimeError):
    """A policy assessment cannot reserve another bounded LLM request."""


@dataclass
class PolicyCallBudget:
    """Per-controller reservation and actual usage; never shared across requests.

    UTF-8 bytes plus message overhead is a conservative input-token envelope,
    independent of the provider tokenizer. Failed calls retain reservations.
    """
    token_limit: int
    deadline: float
    reserved_tokens: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def reserve(self, prompt: Any, output_cap: int) -> None:
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("policy deadline")
        messages = prompt if isinstance(prompt, (list, tuple)) else [prompt]
        size = sum(len(str(getattr(item, "content", item)).encode("utf-8")) + 128 for item in messages)
        reservation = size + output_cap
        if self.reserved_tokens + reservation > self.token_limit:
            raise PolicyBudgetExceeded("policy_llm_token_budget")
        self.reserved_tokens += reservation


policy_call_budget: ContextVar[PolicyCallBudget | None] = ContextVar("policy_call_budget", default=None)


_OUTPUT_TOKEN_DEFAULTS = {
    "intent_router": 1200,
    "after_sales_planner": 1200,
    "query_refine": 600,
    "policy_evidence": 1600,
    "faithfulness_check": 800,
    "faithfulness_repair": 1800,
    "history_summary": 1000,
    "chitchat": 256,
    "reply": 800,
}


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    """Runtime configuration for the isolated OpenAI-compatible judge."""

    api_key: str
    base_url: str
    model: str
    timeout_seconds: float
    max_retries: int
    max_tokens: int
    json_mode: bool
    credential_source: str


class JudgeUnavailable(RuntimeError):
    """The configured judge endpoint could not produce a usable response."""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def get_judge_config() -> JudgeConfig:
    """Return judge-only settings and refuse to reuse the runtime LLM config.

    Evaluation requires a judge that is configured independently of the agent
    under test.  Falling back to ``LLM_*`` silently turns evaluation into
    self-assessment, so a missing ``JUDGE_API_KEY`` is a hard failure.
    """

    judge_key = os.getenv("JUDGE_API_KEY")
    if not judge_key:
        raise JudgeUnavailable(
            "No independent judge credential configured. "
            "Set JUDGE_API_KEY (and optionally JUDGE_BASE_URL/JUDGE_MODEL); "
            "LLM_* credentials are deliberately not accepted for judging."
        )
    base_url = (
        os.getenv("JUDGE_BASE_URL") or "https://api.openai.com/v1"
    ).strip()
    model = (os.getenv("JUDGE_MODEL") or "gpt-4o-mini").strip()
    return JudgeConfig(
        api_key=judge_key,
        base_url=base_url,
        model=model,
        timeout_seconds=float(os.getenv("JUDGE_TIMEOUT_SECONDS", "20")),
        max_retries=int(os.getenv("JUDGE_MAX_RETRIES", "1")),
        # Reasoning-compatible gateways count hidden reasoning against this
        # budget.  A 1k cap can therefore stop after reasoning and expose only
        # `{}` even though the visible verdict is tiny (DeepSeek commonly uses
        # 2k+ reasoning tokens on ambiguous multi-turn cases).
        max_tokens=int(os.getenv("JUDGE_MAX_TOKENS", "3000")),
        json_mode=_env_bool("JUDGE_JSON_MODE", True),
        credential_source="JUDGE_API_KEY",
    )


def judge_config_status() -> dict[str, Any]:
    """Return a deliberately secret-free configuration summary."""

    try:
        config = get_judge_config()
    except JudgeUnavailable as exc:
        return {"configured": False, "error": str(exc)}
    parsed = urlparse(config.base_url)
    return {
        "configured": True,
        "model": config.model,
        "endpoint_host": parsed.netloc or parsed.path.split("/")[0],
        "timeout_seconds": config.timeout_seconds,
        "max_retries": config.max_retries,
        "max_tokens": config.max_tokens,
        "json_mode": config.json_mode,
        "credential_source": config.credential_source,
    }


def agent_model_status() -> dict[str, Any]:
    """Return the runtime Agent model configuration without credentials."""

    base_url = (os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1").strip()
    parsed = urlparse(base_url)
    return {
        "configured": bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")),
        "model": os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
        "endpoint_host": parsed.netloc or parsed.path.split("/")[0],
        "thinking_enabled": _env_bool("LLM_THINKING", False),
    }


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _judge_response_text(data: dict[str, Any]) -> str:
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise JudgeUnavailable("Judge response has no choices[0].message") from exc
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    # Some reasoning-model-compatible gateways put the final JSON in
    # reasoning_content while returning an empty content field.
    text = str(content or message.get("reasoning_content") or "").strip()
    if not text:
        raise JudgeUnavailable("Judge response content is empty")
    return text


def invoke_judge(
    prompt: str,
    *,
    system: str | None = None,
    thinking: bool | None = None,
    max_tokens: int | None = None,
) -> str:
    """Call an OpenAI-compatible judge with bounded output and JSON mode.

    This intentionally uses the small Chat Completions wire contract instead
    of the agent's LangChain model. It avoids agent tool/reasoning adapters and
    keeps judge latency, retries and output size independently bounded.
    """

    config = get_judge_config()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens if max_tokens is not None else config.max_tokens,
        "stream": False,
    }
    if thinking is not None:
        payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
    if config.json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    url = _chat_completions_url(config.base_url)
    last_error: Exception | None = None
    for attempt in range(config.max_retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=config.timeout_seconds,
            )
            if response.status_code >= 400:
                detail = ""
                try:
                    body = response.json()
                    detail = str((body.get("error") or {}).get("message") or "")
                except (ValueError, TypeError, AttributeError):
                    detail = ""
                raise JudgeUnavailable(
                    f"Judge HTTP {response.status_code}: {detail[:240] or 'request failed'}"
                )
            return _judge_response_text(response.json())
        except (requests.RequestException, ValueError, JudgeUnavailable) as exc:
            last_error = exc
            if attempt < config.max_retries:
                time.sleep(min(0.25 * (2**attempt), 1.0))
    raise JudgeUnavailable(
        f"Judge request failed after {config.max_retries + 1} attempt(s): {last_error}"
    ) from last_error


def invoke_judge_stream(
    prompt: str,
    *,
    system: str | None = None,
    thinking: bool | None = None,
    max_tokens: int | None = None,
) -> str:
    """Stream one OpenAI-compatible Judge response and return joined content."""

    config = get_judge_config()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens if max_tokens is not None else config.max_tokens,
        "stream": True,
    }
    if thinking is not None:
        payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
    if config.json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    chunks: list[str] = []
    pending_event = ""

    def consume_event(fragment: str) -> bool:
        nonlocal pending_event
        if fragment == "[DONE]":
            return True
        candidate = pending_event + fragment
        try:
            event = json.loads(candidate)
        except json.JSONDecodeError:
            # Some compatible gateways split one SSE JSON object across
            # multiple physical ``data:`` lines, including inside a content
            # string. Joining without an invented newline restores the wire
            # object while preserving the model's escaped JSON content.
            pending_event = candidate
            return False
        pending_event = ""
        choices = event.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            content = delta.get("content") or delta.get("reasoning_content")
            if content:
                chunks.append(str(content))
        return False

    try:
        with requests.post(
            _chat_completions_url(config.base_url),
            headers=headers,
            json=payload,
            stream=True,
            timeout=(min(20.0, config.timeout_seconds), config.timeout_seconds),
        ) as response:
            if response.status_code >= 400:
                detail = ""
                try:
                    body = response.json()
                    detail = str((body.get("error") or {}).get("message") or "")
                except (ValueError, TypeError, AttributeError):
                    detail = ""
                raise JudgeUnavailable(
                    f"Judge HTTP {response.status_code}: "
                    f"{detail[:240] or 'stream request failed'}"
                )
            # requests guesses ISO-8859-1 for ``text/event-stream`` when the
            # gateway omits a charset.  That silently turns Chinese Judge
            # evidence into mojibake.  SSE JSON is UTF-8 by specification, so
            # keep bytes on the wire and decode explicitly.
            for raw_line in response.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                if isinstance(raw_line, bytes):
                    raw_line = raw_line.decode("utf-8")
                line = raw_line.strip()
                if line.startswith("data:"):
                    event_text = line[5:].strip()
                elif pending_event:
                    # A few OpenAI-compatible gateways pretty-print a single
                    # SSE JSON event across physical lines but only prefix the
                    # first line with ``data:``.  Treat subsequent lines as
                    # continuations of the incomplete event.
                    event_text = line
                else:
                    continue
                if consume_event(event_text):
                    break
    except (requests.RequestException, ValueError, JudgeUnavailable) as exc:
        raise JudgeUnavailable(f"Judge stream request failed: {exc}") from exc
    if pending_event:
        raise JudgeUnavailable("Judge stream ended with an incomplete SSE event")
    text = "".join(chunks).strip()
    if not text:
        raise JudgeUnavailable("Judge stream response content is empty")
    return text


def agent_max_tokens(prompt_kind: str) -> int:
    """Return the configurable output cap for one runtime prompt scene."""

    normalized = re.sub(r"[^A-Za-z0-9]+", "_", prompt_kind).strip("_").upper()
    fallback_key = prompt_kind if prompt_kind in _OUTPUT_TOKEN_DEFAULTS else "reply"
    if not prompt_kind.startswith("reply_") and prompt_kind not in _OUTPUT_TOKEN_DEFAULTS:
        fallback_key = "reply"
    raw = os.getenv(f"LLM_MAX_TOKENS_{normalized}")
    return max(1, int(raw or _OUTPUT_TOKEN_DEFAULTS[fallback_key]))


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def extract_llm_usage(response: Any) -> dict[str, int]:
    """Normalize LangChain/OpenAI/Qwen token and cache usage metadata."""

    usage = _mapping(getattr(response, "usage_metadata", None))
    metadata = _mapping(getattr(response, "response_metadata", None))
    token_usage = _mapping(metadata.get("token_usage") or metadata.get("usage"))
    input_details = _mapping(
        usage.get("input_token_details") or usage.get("prompt_token_details")
    )
    prompt_details = _mapping(
        token_usage.get("prompt_tokens_details")
        or token_usage.get("prompt_token_details")
    )

    def first_int(*values: Any) -> int:
        for value in values:
            if value is not None:
                try:
                    return max(0, int(value))
                except (TypeError, ValueError):
                    continue
        return 0

    input_tokens = first_int(
        usage.get("input_tokens"),
        usage.get("prompt_tokens"),
        token_usage.get("prompt_tokens"),
        token_usage.get("input_tokens"),
    )
    output_tokens = first_int(
        usage.get("output_tokens"),
        usage.get("completion_tokens"),
        token_usage.get("completion_tokens"),
        token_usage.get("output_tokens"),
    )
    total_tokens = first_int(
        usage.get("total_tokens"),
        token_usage.get("total_tokens"),
        input_tokens + output_tokens,
    )
    cached_tokens = first_int(
        usage.get("cached_tokens"),
        usage.get("cache_read_input_tokens"),
        input_details.get("cache_read"),
        input_details.get("cached_tokens"),
        token_usage.get("cached_tokens"),
        prompt_details.get("cached_tokens"),
        prompt_details.get("cache_read"),
    )
    cache_creation_tokens = first_int(
        usage.get("cache_creation_input_tokens"),
        usage.get("cache_write_tokens"),
        input_details.get("cache_creation"),
        input_details.get("cache_write"),
        token_usage.get("cache_creation_input_tokens"),
        token_usage.get("cache_write_tokens"),
        prompt_details.get("cache_creation"),
        prompt_details.get("cache_write"),
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": min(input_tokens, cached_tokens),
        "cache_creation_tokens": min(input_tokens, cache_creation_tokens),
        "cache_miss_tokens": max(0, input_tokens - cached_tokens),
    }


def _estimated_cost(usage: dict[str, int]) -> float | None:
    rates: dict[str, float | None] = {}
    for key, env_name in {
        "input": "LLM_INPUT_COST_PER_1M",
        "cached": "LLM_CACHED_INPUT_COST_PER_1M",
        "creation": "LLM_CACHE_CREATION_COST_PER_1M",
        "output": "LLM_OUTPUT_COST_PER_1M",
    }.items():
        raw = os.getenv(env_name)
        rates[key] = float(raw) if raw not in (None, "") else None
    if all(value is None for value in rates.values()):
        return None
    input_rate = rates["input"] or 0.0
    cached_rate = rates["cached"] if rates["cached"] is not None else input_rate
    creation_rate = (
        rates["creation"] if rates["creation"] is not None else input_rate
    )
    output_rate = rates["output"] or 0.0
    cached = usage["cached_tokens"]
    creation = min(
        usage["cache_creation_tokens"],
        max(0, usage["input_tokens"] - cached),
    )
    ordinary = max(0, usage["input_tokens"] - cached - creation)
    return round(
        (
            ordinary * input_rate
            + cached * cached_rate
            + creation * creation_rate
            + usage["output_tokens"] * output_rate
        )
        / 1_000_000,
        10,
    )


def invoke_with_telemetry(
    model: Any,
    prompt: Any,
    *,
    prompt_kind: str,
    prompt_version: str = "v1",
) -> Any:
    """Invoke a prepared model and persist best-effort usage/cache telemetry.

    ``prompt`` is normally a string.  It may also be a prepared message list
    (see ``prompt_cache.build_cached_messages``), which keeps the stable part
    of the prompt byte-identical across turns so provider prefix caches can
    hit.  Callers doing that should pass ``prompt_version`` from
    ``prompt_cache.prefix_fingerprint`` so cache hits are attributable to a
    specific prefix revision.
    """

    budget = policy_call_budget.get()
    if budget is not None:
        budget.reserve(prompt, agent_max_tokens(prompt_kind))
    started = time.perf_counter()
    base_url = (os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1").strip()
    parsed = urlparse(base_url)
    common = {
        "prompt_kind": prompt_kind,
        "prompt_version": prompt_version,
        "provider": parsed.netloc or parsed.path.split("/")[0],
        "model": os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
        "retry_count": 0,
    }

    def persist(payload: dict[str, Any]) -> None:
        if budget is not None:
            budget.events.append(payload)
        try:
            from .database import record_llm_usage

            record_llm_usage(payload)
        except Exception as exc:
            LOGGER.warning("LLM usage telemetry write failed: %s", exc)

    try:
        if isinstance(prompt, (list, tuple)):
            response = model.invoke(list(prompt))
        else:
            response = model.invoke(prompt)
    except Exception as exc:
        persist(
            {
                **common,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "status": "error",
                "error_type": type(exc).__name__,
            }
        )
        raise
    usage = extract_llm_usage(response)
    persist(
        {
            **common,
            **usage,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "status": "success",
            "error_type": "",
            "estimated_cost": _estimated_cost(usage),
        }
    )
    return response


def get_model(
    temperature: float = 0.7,
    *,
    json_mode: bool = False,
    max_tokens: int | None = None,
):
    """返回 LLM 实例。

    DeepSeek 官方 API 用 ChatDeepSeek，其余 OpenAI 兼容接口用 ChatOpenAI。
    json_mode=True 时通过 OpenAI 兼容的 response_format 强制 JSON Object 输出
    （保证合法 JSON，不保证字段结构），仅建议对 prompt 明确要求 JSON 的调用点开启，
    如意图分析/查询归一；模型不支持时（会 400）由调用方捕获并回退。
    注意 thinking 模式的一个硬约束：带 tool_calls 的 assistant 消息回传时必须附
    reasoning_content，否则 400（The reasoning_content ... must be passed back）。
    实测确认 langchain-openai 与 langchain-deepseek 1.1.0 序列化时都会丢弃该字段，
    真正保证回传的是 agent.py 头部的 monkey-patch（_convert_message_to_dict 注入
    reasoning_content）；这里用官方类仅保证 DeepSeek 官方接口的接入兼容性。
    """
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未配置 LLM_API_KEY。请复制 .env.example 为 .env 并填入你的 API Key。"
        )
    base_url = os.getenv("LLM_BASE_URL")
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "1"))
    budget = policy_call_budget.get()
    if budget is not None:
        timeout = max(0.01, min(timeout, budget.deadline - time.perf_counter()))
        max_retries = 0
    deadline = call_deadline.get()
    if deadline is not None:
        timeout = max(0.01, min(timeout, deadline - time.perf_counter()))
        max_retries = 0
    thinking = _env_bool("LLM_THINKING", False)
    if _HAS_DEEPSEEK and base_url and "deepseek" in base_url.lower():
        kwargs: dict[str, Any] = {
            "model": model_name,
            "api_key": api_key,
            "temperature": temperature,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatDeepSeek(
            **kwargs,
        )
    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": temperature,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs, extra_body={
        "thinking": {"type": "enabled" if thinking else "disabled"}
    })
