"""Tests for the FinishReasonWatcher observability helper."""

from __future__ import annotations

from types import SimpleNamespace

from pydantic_ai_colony import FinishReasonWatcher
from pydantic_ai_colony.observability import _extract_finish_reasons


def _response(
    finish_reason: str | None = None,
    *,
    provider_details: dict | None = None,
    kind: str = "response",
) -> SimpleNamespace:
    """Build a minimal ModelResponse-shaped object."""
    return SimpleNamespace(
        kind=kind,
        finish_reason=finish_reason,
        provider_details=provider_details,
    )


def _user_prompt() -> SimpleNamespace:
    """Build a minimal user-prompt-shaped object — not a model response."""
    return SimpleNamespace(kind="request", finish_reason=None, provider_details=None)


def _result_with_messages(*messages) -> SimpleNamespace:
    """Build something with an .all_messages() method."""
    return SimpleNamespace(all_messages=lambda: list(messages))


# ── _extract_finish_reasons ────────────────────────────────────────


class TestExtractFinishReasons:
    def test_top_level_attribute_preferred(self):
        msgs = [_response("stop")]
        assert _extract_finish_reasons(msgs) == ["stop"]

    def test_provider_details_fallback(self):
        msgs = [_response(None, provider_details={"finish_reason": "length"})]
        assert _extract_finish_reasons(msgs) == ["length"]

    def test_top_level_takes_priority_over_provider_details(self):
        msgs = [_response("stop", provider_details={"finish_reason": "length"})]
        assert _extract_finish_reasons(msgs) == ["stop"]

    def test_stop_reason_alias_in_provider_details(self):
        msgs = [_response(None, provider_details={"stop_reason": "length"})]
        assert _extract_finish_reasons(msgs) == ["length"]

    def test_user_prompts_skipped(self):
        msgs = [_user_prompt(), _response("length"), _user_prompt()]
        assert _extract_finish_reasons(msgs) == ["length"]

    def test_missing_metadata_returns_empty(self):
        msgs = [_response(None)]
        assert _extract_finish_reasons(msgs) == []

    def test_none_input_returns_empty(self):
        assert _extract_finish_reasons(None) == []

    def test_non_iterable_returns_empty(self):
        assert _extract_finish_reasons(42) == []

    def test_multiple_responses_all_collected(self):
        msgs = [_response("stop"), _response("length"), _response("tool_call")]
        assert _extract_finish_reasons(msgs) == ["stop", "length", "tool_call"]


# ── FinishReasonWatcher ────────────────────────────────────────────


class TestFinishReasonWatcher:
    def test_initial_state(self):
        w = FinishReasonWatcher()
        assert w.last_finish_reason is None
        assert w.length_count == 0
        assert w.total_count == 0

    def test_observe_run_result_with_all_messages(self):
        w = FinishReasonWatcher(log_level=None)
        result = _result_with_messages(_user_prompt(), _response("length"))
        w.observe(result)
        assert w.last_finish_reason == "length"
        assert w.length_count == 1
        assert w.total_count == 1

    def test_observe_raw_message_iterable(self):
        w = FinishReasonWatcher(log_level=None)
        w.observe([_response("stop"), _response("length")])
        assert w.last_finish_reason == "length"
        assert w.length_count == 1
        assert w.total_count == 2

    def test_observe_none_is_noop(self):
        w = FinishReasonWatcher(log_level=None)
        w.observe(None)
        assert w.total_count == 0
        assert w.last_finish_reason is None

    def test_warning_emitted_on_length(self, caplog):
        w = FinishReasonWatcher()
        with caplog.at_level("WARNING", logger="pydantic_ai_colony"):
            w.observe([_response("length")])
        assert any("finish_reason=length" in record.message for record in caplog.records)

    def test_no_warning_on_stop(self, caplog):
        w = FinishReasonWatcher()
        with caplog.at_level("WARNING", logger="pydantic_ai_colony"):
            w.observe([_response("stop")])
        assert not any("finish_reason=length" in record.message for record in caplog.records)

    def test_log_level_none_silences_warning(self, caplog):
        w = FinishReasonWatcher(log_level=None)
        with caplog.at_level("WARNING", logger="pydantic_ai_colony"):
            w.observe([_response("length")])
        assert w.length_count == 1
        assert not any("finish_reason=length" in record.message for record in caplog.records)

    def test_multiple_observes_track_last(self):
        w = FinishReasonWatcher(log_level=None)
        w.observe([_response("stop")])
        w.observe([_response("length")])
        w.observe([_response("stop")])
        assert w.last_finish_reason == "stop"
        assert w.length_count == 1
        assert w.total_count == 3

    def test_reset_clears_state(self):
        w = FinishReasonWatcher(log_level=None)
        w.observe([_response("length"), _response("stop")])
        w.reset()
        assert w.last_finish_reason is None
        assert w.length_count == 0
        assert w.total_count == 0

    def test_observe_run_result_where_all_messages_takes_args(self):
        # Some pydantic-ai versions may take args on all_messages; make sure
        # we fall back to treating the result itself as iterable when the
        # bare call fails.
        class StubResult:
            def __iter__(self):
                return iter([_response("length")])

            def all_messages(self, required_arg):  # noqa: ARG002
                raise TypeError("missing arg")

        w = FinishReasonWatcher(log_level=None)
        w.observe(StubResult())
        assert w.last_finish_reason == "length"
        assert w.length_count == 1

    def test_provider_details_path_observed(self):
        w = FinishReasonWatcher(log_level=None)
        msg = _response(None, provider_details={"finish_reason": "length"})
        w.observe([msg])
        assert w.length_count == 1
        assert w.last_finish_reason == "length"
