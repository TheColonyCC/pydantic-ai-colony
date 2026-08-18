"""Our own truncation has to announce itself.

`_looks_truncated` exists in this package because the *server* cuts DM previews
and sends no flag, and 0.9.0's changelog is titled "A preview is not a message".
The same file then did exactly that to its own consumers, in seven places, with
a bare ``[:max_body]``.

On 2026-08-18 that cost something real: a downstream agent was handed a 1,699
character post cut to 1,500 by a caller of this library, correctly saw that the
text stopped mid-sentence, and said in public that the *author* had posted it
that way. It was truthful about the bytes it received. Nothing in the payload
told it the omission was ours.

Every test here is paired with a control, because a truncation marker that is
always present is as useless as one that is never present.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from pydantic_ai_colony import ColonyReadOnlyToolset
from pydantic_ai_colony.toolset import DEFAULT_MAX_BODY, _excerpt

LONG = "x" * 1699
SHORT = "a complete short body."


def _client(**overrides: Any) -> MagicMock:
    c = MagicMock()
    c.get_posts.return_value = {"items": [{"id": "p1", "title": "t", "body": LONG, "author": {"username": "u"}}]}
    c.get_post.return_value = {"id": "p1", "title": "t", "body": LONG, "author": {"username": "u"}}
    c.get_user.return_value = {"id": "u1", "username": "u", "bio": LONG}
    c.directory.return_value = {"users": [{"id": "u1", "username": "u", "bio": LONG}]}
    c.iter_comments.return_value = iter([{"id": "c1", "body": LONG, "author": {"username": "u"}}])
    for k, v in overrides.items():
        getattr(c, k).return_value = v
    return c


class TestExcerpt:
    def test_long_text_is_cut_and_says_so(self) -> None:
        out, cut = _excerpt(LONG, 1500)
        assert cut is True
        assert "cut by pydantic-ai-colony" in out
        assert "at 1500 of 1699 chars" in out

    def test_short_text_is_returned_byte_identical(self) -> None:
        """The control. A marker on everything carries no information."""
        out, cut = _excerpt(SHORT, 1500)
        assert out == SHORT
        assert cut is False
        assert "truncated" not in out

    def test_exactly_at_the_limit_is_not_cut(self) -> None:
        exact = "y" * 1500
        out, cut = _excerpt(exact, 1500)
        assert out == exact and cut is False

    def test_one_over_the_limit_is_cut(self) -> None:
        out, cut = _excerpt("y" * 1501, 1500)
        assert cut is True and "at 1500 of 1501 chars" in out

    def test_empty_text_is_not_cut(self) -> None:
        assert _excerpt("", 1500) == ("", False)

    def test_the_marker_blames_us_not_the_author(self) -> None:
        """The whole point. A model must not conclude the source is malformed."""
        out, _ = _excerpt(LONG, 1500)
        assert "OUR cut, not the author's" in out
        assert "the source is not malformed" in out

    def test_full_text_hint_is_included_when_given(self) -> None:
        out, _ = _excerpt(LONG, 1500, full_text="colony_get_post(post_id)")
        assert "Call colony_get_post(post_id) for the full text." in out

    def test_no_hint_when_there_is_no_such_tool(self) -> None:
        """Comments have no untruncated tool; do not invent advice."""
        out, _ = _excerpt(LONG, 1500)
        assert "Call " not in out


class TestToolsAnnounceTheirOwnCuts:
    @pytest.mark.asyncio
    async def test_get_posts_flags_and_marks(self) -> None:
        ts = ColonyReadOnlyToolset(_client())
        r = await ts.tools["colony_get_posts"].function()
        post = r["posts"][0]
        assert post["body_is_truncated"] is True
        assert "cut by pydantic-ai-colony" in post["body"]

    @pytest.mark.asyncio
    async def test_get_posts_body_is_not_the_bare_slice(self) -> None:
        """Mutation arm: reverting to ``body[:max_body]`` fails here."""
        ts = ColonyReadOnlyToolset(_client())
        r = await ts.tools["colony_get_posts"].function()
        assert r["posts"][0]["body"] != LONG[:DEFAULT_MAX_BODY]

    @pytest.mark.asyncio
    async def test_short_body_is_untouched_and_unflagged(self) -> None:
        """The control, through the tool rather than the helper."""
        ts = ColonyReadOnlyToolset(
            _client(get_posts={"items": [{"id": "p1", "title": "t", "body": SHORT, "author": {}}]})
        )
        r = await ts.tools["colony_get_posts"].function()
        assert r["posts"][0]["body"] == SHORT
        assert r["posts"][0]["body_is_truncated"] is False

    @pytest.mark.asyncio
    async def test_comments_are_flagged(self) -> None:
        ts = ColonyReadOnlyToolset(_client())
        r = await ts.tools["colony_get_comments"].function(post_id="p1")
        assert r["comments"][0]["body_is_truncated"] is True

    @pytest.mark.asyncio
    async def test_directory_bio_is_flagged(self) -> None:
        ts = ColonyReadOnlyToolset(_client())
        r = await ts.tools["colony_directory"].function()
        assert r["users"][0]["bio_is_truncated"] is True


class TestTheAdviceIsReal:
    """The marker tells the model to call a tool. That tool must actually help."""

    @pytest.mark.asyncio
    async def test_get_post_returns_the_untruncated_body(self) -> None:
        ts = ColonyReadOnlyToolset(_client())
        r = await ts.tools["colony_get_post"].function(post_id="p1")
        assert r["body"] == LONG
        assert "cut by pydantic-ai-colony" not in r["body"]

    @pytest.mark.asyncio
    async def test_get_user_returns_the_untruncated_bio(self) -> None:
        ts = ColonyReadOnlyToolset(_client())
        r = await ts.tools["colony_get_user"].function(user_id="u1")
        assert r["bio"] == LONG
