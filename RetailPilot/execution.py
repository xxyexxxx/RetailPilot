"""Fixed daemon workers, bounded admission and propagated real deadlines.

Timed-out running calls retain capacity until the underlying client returns.
Queued calls can be cancelled; overload fails fast instead of spawning threads.
Separate policy/tool pools avoid nested-call starvation.
"""

from __future__ import annotations

import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from contextvars import ContextVar, copy_context
from queue import Queue
from threading import BoundedSemaphore, Lock, Thread
from typing import Any

call_deadline: ContextVar[float | None] = ContextVar("call_deadline", default=None)


class BoundedPool:
    def __init__(self, name: str, workers: int, capacity: int):
        self.name, self.workers = name, workers
        self._slots = BoundedSemaphore(capacity)
        self._queue: Queue = Queue(maxsize=capacity)
        self._lock = Lock()
        self._started = False

    def _worker(self):
        while True:
            future, context, call, deadline = self._queue.get()
            try:
                if future.set_running_or_notify_cancel():

                    def invoke(call=call, deadline=deadline):
                        inherited = call_deadline.get()
                        effective = min(inherited, deadline) if inherited else deadline
                        token = call_deadline.set(effective)
                        try:
                            if time.perf_counter() >= effective:
                                raise TimeoutError("execution deadline exhausted")
                            return call()
                        finally:
                            call_deadline.reset(token)

                    try:
                        future.set_result(context.run(invoke))
                    except BaseException as exc:  # noqa: BLE001 - worker boundary must settle Future
                        future.set_exception(exc)
            finally:
                self._slots.release()
                self._queue.task_done()

    def submit(self, call, *, deadline: float) -> Future:
        if deadline <= time.perf_counter():
            raise TimeoutError("execution deadline exhausted")
        if not self._slots.acquire(blocking=False):
            raise RuntimeError("execution capacity exhausted")
        with self._lock:
            if not self._started:
                for index in range(self.workers):
                    Thread(
                        target=self._worker, name=f"{self.name}-{index}", daemon=True
                    ).start()
                self._started = True
        future: Future = Future()
        self._queue.put_nowait((future, copy_context(), call, deadline))
        return future

    def call(self, call, deadline: float) -> Any:
        future = self.submit(call, deadline=deadline)
        try:
            return future.result(timeout=max(0, deadline - time.perf_counter()))
        except FutureTimeout as exc:
            future.cancel()
            raise TimeoutError("execution deadline exhausted") from exc


policy_pool = BoundedPool("policy", 4, 12)
tool_pool = BoundedPool("tool", 8, 24)
chat_pool = BoundedPool("chat", 4, 8)
