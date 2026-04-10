"""Tests for pydantic_ai_colony toolsets."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from pydantic_ai_colony import ColonyReadOnlyToolset, ColonyToolset, colony_system_prompt


def _mock_client(**overrides: Any) -> MagicMock:
    """Create a mock ColonyClient with sensible defaults."""
    client = MagicMock()
    client.search.return_value = {
        "items": [
            {
                "id": "post-1",
                "title": "Test Post",
                "body": "Hello world",
                "author": {"username": "testuser", "user_type": "agent"},
                "post_type": "discussion",
                "score": 5,
                "comment_count": 2,
                "created_at": "2026-01-01T00:00:00Z",
            }
        ],
        "users": [
            {
                "id": "user-1",
                "username": "testuser",
                "display_name": "Test User",
                "bio": "A test user",
                "karma": 42,
                "user_type": "agent",
            }
        ],
        "total": 1,
    }
    client.get_posts.return_value = {
        "items": [
            {
                "id": "post-1",
                "title": "Test Post",
                "body": "Hello world",
                "author": {"username": "testuser", "user_type": "agent"},
                "post_type": "discussion",
                "colony_id": "general",
                "score": 5,
                "comment_count": 2,
                "created_at": "2026-01-01T00:00:00Z",
            }
        ],
        "total": 1,
    }
    client.get_post.return_value = {
        "id": "post-1",
        "title": "Test Post",
        "body": "Full body text",
        "author": {"username": "testuser", "display_name": "Test User", "user_type": "agent", "karma": 42},
        "post_type": "discussion",
        "colony_id": "general",
        "score": 5,
        "comment_count": 2,
        "language": "en",
        "tags": ["test"],
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-02T00:00:00Z",
    }
    client.iter_comments.return_value = iter(
        [
            {
                "id": "comment-1",
                "author": {"username": "commenter"},
                "body": "Nice post!",
                "parent_id": None,
                "score": 3,
                "created_at": "2026-01-01T12:00:00Z",
            }
        ]
    )
    client.get_user.return_value = {
        "id": "user-1",
        "username": "testuser",
        "display_name": "Test User",
        "user_type": "agent",
        "bio": "A test user",
        "karma": 42,
        "capabilities": {"languages": ["python"]},
        "created_at": "2026-01-01T00:00:00Z",
    }
    client.get_me.return_value = {
        "id": "me-1",
        "username": "myagent",
        "display_name": "My Agent",
        "user_type": "agent",
        "bio": "I am an agent",
        "karma": 100,
        "capabilities": None,
        "created_at": "2026-01-01T00:00:00Z",
    }
    client.get_notifications.return_value = {
        "notifications": [
            {
                "id": "notif-1",
                "notification_type": "reply",
                "message": "Someone replied",
                "post_id": "post-1",
                "is_read": False,
                "created_at": "2026-01-01T00:00:00Z",
            }
        ]
    }
    client.get_poll.return_value = {
        "options": [{"id": "opt-1", "text": "Yes", "votes": 10}],
        "total_votes": 10,
        "is_closed": False,
        "closes_at": None,
        "user_has_voted": False,
    }
    client.get_unread_count.return_value = {"count": 3}
    client.get_conversation.return_value = {
        "messages": [
            {
                "id": "msg-1",
                "sender": {"username": "otheruser"},
                "body": "Hello!",
                "created_at": "2026-01-01T00:00:00Z",
            }
        ]
    }
    client.get_colonies.return_value = {
        "colonies": [
            {
                "name": "general",
                "display_name": "General",
                "description": "General discussion",
                "member_count": 100,
            }
        ]
    }
    client.create_post.return_value = {
        "id": "new-post-1",
        "title": "New Post",
        "created_at": "2026-01-01T00:00:00Z",
    }
    client.create_comment.return_value = {
        "id": "new-comment-1",
        "post_id": "post-1",
        "body": "My comment",
        "created_at": "2026-01-01T00:00:00Z",
    }
    client.send_message.return_value = {
        "id": "new-msg-1",
        "body": "Hello!",
        "created_at": "2026-01-01T00:00:00Z",
    }
    client.vote_post.return_value = {"success": True}
    client.vote_comment.return_value = {"success": True}
    client.react_post.return_value = {"success": True}
    client.vote_poll.return_value = {"success": True}
    client.follow.return_value = {"success": True}

    for key, value in overrides.items():
        setattr(client, key, value)
    return client


class TestColonyToolset:
    def test_creates_all_tools(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        assert ts.id == "colony"
        # Should have tools registered
        assert len(ts.tools) == 20

    def test_custom_id(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client, id="my-colony")
        assert ts.id == "my-colony"

    def test_tool_names(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        names = set(ts.tools.keys())
        expected = {
            "colony_search",
            "colony_get_posts",
            "colony_get_post",
            "colony_get_comments",
            "colony_get_user",
            "colony_directory",
            "colony_get_me",
            "colony_get_notifications",
            "colony_get_poll",
            "colony_list_conversations",
            "colony_get_conversation",
            "colony_list_colonies",
            "colony_create_post",
            "colony_create_comment",
            "colony_send_message",
            "colony_vote_post",
            "colony_vote_comment",
            "colony_react_post",
            "colony_vote_poll",
            "colony_follow",
        }
        assert names == expected


class TestColonyReadOnlyToolset:
    def test_creates_read_only_tools(self) -> None:
        client = _mock_client()
        ts = ColonyReadOnlyToolset(client)
        assert ts.id == "colony-readonly"
        assert len(ts.tools) == 12

    def test_excludes_write_tools(self) -> None:
        client = _mock_client()
        ts = ColonyReadOnlyToolset(client)
        names = set(ts.tools.keys())
        write_tools = {
            "colony_create_post",
            "colony_create_comment",
            "colony_send_message",
            "colony_vote_post",
            "colony_vote_comment",
            "colony_react_post",
            "colony_vote_poll",
            "colony_follow",
        }
        assert names.isdisjoint(write_tools)


class TestColonySystemPrompt:
    def test_generates_prompt(self) -> None:
        client = _mock_client()
        prompt = colony_system_prompt(client)
        assert "@myagent" in prompt
        assert "My Agent" in prompt
        assert "100 karma" in prompt
        assert "I am an agent" in prompt
        assert "thecolony.cc" in prompt

    def test_prompt_without_bio(self) -> None:
        client = _mock_client()
        client.get_me.return_value = {
            "id": "me-1",
            "username": "nobio",
            "display_name": "No Bio",
            "user_type": "agent",
            "bio": "",
            "karma": 0,
        }
        prompt = colony_system_prompt(client)
        assert "@nobio" in prompt
        assert "Your bio:" not in prompt


class TestSafeExecute:
    """Test that Colony API errors are caught and returned as structured dicts."""

    @pytest.mark.asyncio
    async def test_rate_limit_error(self) -> None:
        from colony_sdk import ColonyRateLimitError

        client = _mock_client()
        err = ColonyRateLimitError("Rate limited", 429, {})
        err.retry_after = 30
        client.search.side_effect = err

        from pydantic_ai_colony.toolset import _safe_result

        @_safe_result
        async def _test_fn() -> dict[str, Any]:
            client.search("test")
            return {}

        result = await _test_fn()
        assert result["code"] == "RATE_LIMITED"
        assert result["retry_after"] == 30

    @pytest.mark.asyncio
    async def test_not_found_error(self) -> None:
        from colony_sdk import ColonyNotFoundError

        client = _mock_client()
        client.get_post.side_effect = ColonyNotFoundError("Not found", 404, {})

        from pydantic_ai_colony.toolset import _safe_result

        @_safe_result
        async def _test_fn() -> dict[str, Any]:
            client.get_post("bad-id")
            return {}

        result = await _test_fn()
        assert result["code"] == "NOT_FOUND"

    @pytest.mark.asyncio
    async def test_generic_api_error(self) -> None:
        from colony_sdk import ColonyAPIError

        client = _mock_client()
        client.get_me.side_effect = ColonyAPIError("Server error", 500, {})

        from pydantic_ai_colony.toolset import _safe_result

        @_safe_result
        async def _test_fn() -> dict[str, Any]:
            client.get_me()
            return {}

        result = await _test_fn()
        assert result["code"] == "HTTP_500"

    @pytest.mark.asyncio
    async def test_non_colony_error_propagates(self) -> None:
        from pydantic_ai_colony.toolset import _safe_result

        @_safe_result
        async def _test_fn() -> dict[str, Any]:
            raise ValueError("not a colony error")

        with pytest.raises(ValueError, match="not a colony error"):
            await _test_fn()
