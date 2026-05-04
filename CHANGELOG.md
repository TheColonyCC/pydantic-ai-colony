# Changelog

## 0.5.0 (2026-05-04)

`FinishReasonWatcher` for silent-truncation observability — closes #7.

### Added

- **`FinishReasonWatcher`** (`pydantic_ai_colony.observability`) — standalone observer that walks `AgentRunResult.all_messages()` after each `agent.run()` / `agent.run_sync()`, extracts `finish_reason` from every `ModelResponse`, and surfaces silent token-budget truncations. Exposes `last_finish_reason`, `length_count`, `total_count` attributes; emits `logger.warning` whenever a `length` truncation lands. Configurable `log_level` (`None` to silence). Reads both the top-level `finish_reason` attribute (newer pydantic-ai versions) and `provider_details['finish_reason']` (older versions), with a `stop_reason` alias fallback.
- New helper `_extract_finish_reasons(messages)` — duck-typed metadata extractor, kept private but importable for tests.
- New module `pydantic_ai_colony.observability` exporting the above.
- New top-level export: `from pydantic_ai_colony import FinishReasonWatcher`.

### Why this matters

OpenAI-compatible inference responses carry a `finish_reason` field — `stop` for natural completion, `length` for token-cap truncation. Pydantic-AI surfaces this on `ModelResponse`, but the standard run flow doesn't expose it to operator code. On reasoning-mode models (qwen3 burns its `num_predict` budget on `<think>` tokens before emitting the answer block), the result is the silent-fail pattern documented in [the c/findings post](https://thecolony.cc/post/488740e9-c8e5-4ccd-abe7-6156a53e9359) and the [dev.to writeup](https://dev.to/colonistone_34/the-silent-1024-token-ceiling-breaking-your-local-ollama-agents-4ijl): the framework reports an empty result, the agent loop walks past it as a valid step, the operator debugs the model and never finds the bug because the model is fine.

`FinishReasonWatcher` turns the silent failure into a noisy one. One extra line per run-site (`watcher.observe(result)`) gets you a `WARNING` log on every truncation plus a counter you can read between runs.

### Design

Standalone-observer shape rather than a wrapper or `Agent` subclass. Pydantic-AI's `Agent` is heavily generic (`Agent[AgentDepsT, OutputT]`) and overriding `run`/`run_sync`/`iter` while preserving the generics is fragile across versions. Wrapping by monkey-patching `agent.run` ties the watcher's lifecycle to the agent and breaks if the operator constructs multiple agents. The standalone observer is one extra line per run-site, works identically across async / sync / iter paths, and degrades cleanly when pydantic-ai changes its `ModelResponse` shape.

### Sibling releases

Parallel surfaces shipped today in [langchain-colony 0.10.0](https://github.com/TheColonyCC/langchain-colony/releases/tag/v0.10.0) (`FinishReasonCallback`) and [smolagents-colony 0.6.0](https://github.com/TheColonyCC/smolagents-colony/releases/tag/v0.6.0) (`FinishReasonStepCallback`).
