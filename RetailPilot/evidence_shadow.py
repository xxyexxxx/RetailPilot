"""Bounded read-only audits; shadow latency cannot hold up the legacy answer."""

from __future__ import annotations

import json
import logging
import time
from contextvars import ContextVar

from .execution import BoundedPool, call_deadline

mode_override = ContextVar("evidence_mode_override", default=None)
audit_futures = ContextVar("evidence_audit_futures", default=None)
_pool = BoundedPool("evidence-shadow", 2, 4)
LOGGER = logging.getLogger(__name__)


def schedule(call, *, task_id):
    def audit():
        token = mode_override.set("unified")
        try:
            result = call()
            verdict = result.get("verdict") or result.get("assessment") or {}
            LOGGER.info(
                "evidence_shadow %s",
                json.dumps(
                    {
                        "task_id": task_id,
                        "next_action": verdict.get("next_action"),
                        "summary": result.get("summary", {}),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            )
            return result
        finally:
            mode_override.reset(token)

    token = call_deadline.set(time.perf_counter() + 30)
    try:
        future = _pool.submit(audit, deadline=call_deadline.get())
        pending = audit_futures.get()
        if pending is not None:
            pending.append(future)
        return {"node": "evidence.shadow", "status": "started", "task_id": task_id}
    except RuntimeError:
        return {"node": "evidence.shadow", "status": "skipped", "task_id": task_id}
    finally:
        call_deadline.reset(token)
