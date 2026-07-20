from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

@dataclass(frozen=True)
class TokenUsage:
    timestamp: float
    tokens: int


class TokenPerMinuteLimiter:
    """
    Thread-safe rolling-window token-per-minute limiter.

    Before making a request, call acquire() with the estimated number
    of input and output tokens. The method blocks until the request fits
    inside the configured rolling 60-second budget.
    """

    def __init__(
        self,
        tokens_per_minute: int,
        window_seconds: float = 60.0,
    ) -> None:
        if tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute must be greater than zero")

        self.tokens_per_minute = tokens_per_minute
        self.window_seconds = window_seconds
        self._usage: Deque[TokenUsage] = deque()
        self._lock = threading.Lock()

    def acquire(self, estimated_tokens: int) -> None:
        if estimated_tokens <= 0:
            return

        if estimated_tokens > self.tokens_per_minute:
            raise ValueError(
                f"Single request requires approximately {estimated_tokens} "
                f"tokens, exceeding the configured limit of "
                f"{self.tokens_per_minute} tokens per minute."
            )

        while True:
            with self._lock:
                now = time.monotonic()
                self._discard_expired_usage(now)

                currently_used = sum(
                    usage.tokens
                    for usage in self._usage
                )

                if (
                    currently_used + estimated_tokens
                    <= self.tokens_per_minute
                ):
                    self._usage.append(
                        TokenUsage(
                            timestamp=now,
                            tokens=estimated_tokens,
                        )
                    )
                    return

                oldest_usage = self._usage[0]

                wait_seconds = (
                    oldest_usage.timestamp
                    + self.window_seconds
                    - now
                )

            time.sleep(max(wait_seconds, 0.1))

    def _discard_expired_usage(self, now: float) -> None:
        cutoff = now - self.window_seconds

        while self._usage and self._usage[0].timestamp <= cutoff:
            self._usage.popleft()


_PROVIDER_TPM_LIMITS = {
    "groq": 12_000,
    "gemini": 30_000,
    "openrouter": 12_000,
    "openai": 30_000,
}


_PROVIDER_LIMITERS = {
    provider: TokenPerMinuteLimiter(tokens_per_minute=limit)
    for provider, limit in _PROVIDER_TPM_LIMITS.items()
}


def estimate_text_tokens(text: str) -> int:
    """
    Conservative provider-independent token estimate.

    English prose is commonly around four characters per token.
    Using three characters per token gives us additional safety.
    """
    if not text:
        return 0

    return max(1, len(text) // 3)


def wait_for_token_capacity(
    *,
    provider: str,
    messages: list[dict[str, str]],
    max_output_tokens: int,
) -> None:
    limiter = _PROVIDER_LIMITERS.get(provider.lower())

    if limiter is None:
        return

    input_tokens = sum(
        estimate_text_tokens(message.get("content", ""))
        for message in messages
    )

    estimated_total_tokens = input_tokens + max_output_tokens

    limiter.acquire(estimated_total_tokens)