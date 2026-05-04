"""Observability helpers for pydantic-ai-colony agents.

Currently exposes one helper: :class:`FinishReasonWatcher`, which detects
silent token-budget truncations on local-Ollama (and other) inference
backends. See https://thecolony.cc/post/488740e9-c8e5-4ccd-abe7-6156a53e9359
for the failure-mode writeup.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("pydantic_ai_colony")


def _extract_finish_reasons(messages: Any) -> list[str]:
    """Pull every ``finish_reason`` from a sequence of pydantic-ai messages.

    Walks the iterable from :meth:`AgentRunResult.all_messages` (or
    :meth:`new_messages`) and reads ``finish_reason`` from each
    ``ModelResponse``. Prefers the top-level attribute (newer
    pydantic-ai versions) and falls back to
    ``provider_details['finish_reason']`` for older releases. Returns
    a flat list of values; empty when no message surfaces it.

    The walker is duck-typed rather than imported-typed so this module
    keeps working across pydantic-ai versions where the
    ``ModelResponse`` import path or shape may shift.
    """
    out: list[str] = []
    if messages is None:
        return out
    try:
        iterator = list(messages)
    except TypeError:
        return out
    for msg in iterator:
        # We only care about model responses (not user prompts or tool
        # returns). pydantic-ai marks these with ``kind='response'``.
        kind = getattr(msg, "kind", None)
        if kind not in (None, "response"):
            continue
        value: str | None = None
        # Prefer the top-level field (newer versions).
        attr = getattr(msg, "finish_reason", None)
        if attr:
            value = str(attr)
        # Fall back to provider_details.
        if value is None:
            details = getattr(msg, "provider_details", None) or {}
            if isinstance(details, dict):
                raw = details.get("finish_reason") or details.get("stop_reason")
                if raw:
                    value = str(raw)
        if value:
            out.append(value)
    return out


class FinishReasonWatcher:
    """Detect silent token-budget truncations from a pydantic-ai agent.

    OpenAI-compatible inference responses include a ``finish_reason``
    field — ``stop`` when the model finished naturally, ``length`` when
    it hit the token cap mid-thought. Pydantic-AI surfaces this on
    :class:`ModelResponse` (top-level attribute and/or
    ``provider_details``), but most agent loops never read it. With
    qwen3 / other reasoning-mode models on a tight ``num_predict``,
    that's the silent-fail pattern documented at
    https://thecolony.cc/post/488740e9-c8e5-4ccd-abe7-6156a53e9359.

    Usage (most ergonomic — observe each run yourself)::

        from pydantic_ai_colony.observability import FinishReasonWatcher

        watcher = FinishReasonWatcher()

        result = await agent.run("...")
        watcher.observe(result)

        if watcher.length_count:
            print(f"hit num_predict {watcher.length_count} time(s)")

    Works with both async (:meth:`Agent.run`) and sync
    (:meth:`Agent.run_sync`) results — :meth:`observe` is sync and
    accepts anything with an ``all_messages()`` method, or directly an
    iterable of pydantic-ai messages.

    Args:
        log_level: Logging level for the warning emitted on ``length``.
            Set to ``None`` to disable logging and only collect counters.
            Defaults to ``logging.WARNING``.
    """

    #: The most recently observed finish_reason, or ``None`` if no
    #: ModelResponse with the field has been observed.
    last_finish_reason: str | None

    #: Count of model responses where ``finish_reason == "length"``.
    length_count: int

    #: Count of all model responses observed (with surfaced finish_reason).
    total_count: int

    def __init__(self, log_level: int | None = logging.WARNING) -> None:
        self.log_level = log_level
        self.last_finish_reason = None
        self.length_count = 0
        self.total_count = 0

    def observe(self, result_or_messages: Any) -> None:
        """Record finish_reason values from a run result or message list.

        Accepts:

        * An :class:`AgentRunResult` (calls ``all_messages()``).
        * A pre-extracted message iterable.
        * ``None`` — no-op.

        Updates :attr:`last_finish_reason`, :attr:`length_count`,
        :attr:`total_count`, and emits :data:`logger.warning` for each
        ``length`` value when ``log_level`` is set.
        """
        if result_or_messages is None:
            return
        all_messages = getattr(result_or_messages, "all_messages", None)
        if callable(all_messages):
            try:
                messages: Any = all_messages()
            except TypeError:
                # Some versions take args; bail to the iterable path.
                messages = result_or_messages
        else:
            messages = result_or_messages
        reasons = _extract_finish_reasons(messages)
        if not reasons:
            return
        self.total_count += len(reasons)
        self.last_finish_reason = reasons[-1]
        for reason in reasons:
            if reason == "length":
                self.length_count += 1
                if self.log_level is not None:
                    logger.log(
                        self.log_level,
                        "LLM finish_reason=length — likely truncated "
                        "mid-thought, consider raising max_tokens / num_predict",
                    )

    def reset(self) -> None:
        """Reset counters and the last-seen reason."""
        self.last_finish_reason = None
        self.length_count = 0
        self.total_count = 0
