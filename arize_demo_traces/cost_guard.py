"""
Simple cost guard that limits LLM calls per generate request.
"""

import threading

MAX_LLM_CALLS = 10


class CostGuard:
    """Thread-safe counter that raises when the LLM call limit is hit."""

    def __init__(self, max_calls: int = MAX_LLM_CALLS):
        self.max_calls = max_calls
        self._count = 0
        self._lock = threading.Lock()

    def check(self) -> None:
        with self._lock:
            if self._count >= self.max_calls:
                raise RuntimeError(
                    f"Cost guard: reached {self.max_calls} LLM calls for this request. "
                    "Stopping to prevent runaway costs."
                )
            self._count += 1

    @property
    def calls_made(self) -> int:
        return self._count

    def reset(self) -> None:
        with self._lock:
            self._count = 0
