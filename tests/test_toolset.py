"""Tests for pydantic_ai_colony toolsets."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from pydantic_ai_colony import ColonyReadOnlyToolset, ColonyToolset, colony_system_prompt
from pydantic_ai_colony.toolset import _safe_result


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
        "author": {
            "username": "testuser",
            "display_name": "Test User",
            "user_type": "agent",
            "karma": 42,
        },
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
    client.list_conversations.return_value = {
        "conversations": [
            {
                "other_user": "otheruser",
                "last_message_at": "2026-01-01T00:00:00Z",
                "last_message_preview": "Hello!",
                "unread_count": 1,
            }
        ]
    }
    client.directory.return_value = {
        "items": [
            {
                "id": "user-1",
                "username": "testuser",
                "display_name": "Test User",
                "user_type": "agent",
                "bio": "A test user",
                "karma": 42,
            }
        ],
        "total": 1,
    }
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
    client.react_comment.return_value = {"success": True}
    client.vote_poll.return_value = {"success": True}
    client.follow.return_value = {"success": True}
    client.unfollow.return_value = {"success": True}
    client.update_post.return_value = {
        "id": "post-1",
        "title": "Updated Title",
        "updated_at": "2026-01-02T00:00:00Z",
    }
    client.delete_post.return_value = {"success": True}
    client.mark_notifications_read.return_value = None
    client.join_colony.return_value = {"success": True}
    client.leave_colony.return_value = {"success": True}
    client.get_notification_count.return_value = {"count": 5}
    client.get_unread_count.return_value = {"count": 3}
    client.iter_posts.return_value = iter(
        [
            {
                "id": "post-1",
                "title": "Test Post",
                "body": "Hello world",
                "author": {"username": "testuser"},
                "post_type": "discussion",
                "colony_id": "general",
                "score": 5,
                "comment_count": 2,
                "created_at": "2026-01-01T00:00:00Z",
            }
        ]
    )

    for key, value in overrides.items():
        setattr(client, key, value)
    return client


# ── Toolset structure tests ──────────────────────────────────────


class TestColonyToolset:
    def test_creates_all_tools(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        assert ts.id == "colony"
        assert len(ts.tools) == 30

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
            "colony_react_comment",
            "colony_vote_poll",
            "colony_follow",
            "colony_unfollow",
            "colony_update_post",
            "colony_delete_post",
            "colony_mark_notifications_read",
            "colony_join_colony",
            "colony_leave_colony",
            "colony_get_notification_count",
            "colony_get_unread_count",
            "colony_iter_posts",
        }
        assert names == expected


class TestColonyReadOnlyToolset:
    def test_creates_read_only_tools(self) -> None:
        client = _mock_client()
        ts = ColonyReadOnlyToolset(client)
        assert ts.id == "colony-readonly"
        assert len(ts.tools) == 15

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
            "colony_react_comment",
            "colony_vote_poll",
            "colony_follow",
            "colony_unfollow",
            "colony_update_post",
            "colony_delete_post",
            "colony_mark_notifications_read",
            "colony_join_colony",
            "colony_leave_colony",
        }
        assert names.isdisjoint(write_tools)


# ── Per-tool execute tests ───────────────────────────────────────


class TestSearchTool:
    @pytest.mark.asyncio
    async def test_calls_sdk_with_all_params(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_search"].function
        result = await fn(query="AI agents", limit=10, post_type="finding", sort="newest")
        client.search.assert_called_once_with("AI agents", limit=10, post_type="finding", sort="newest")
        assert result["posts"][0]["id"] == "post-1"
        assert result["users"][0]["username"] == "testuser"
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_defaults(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_search"].function
        await fn(query="test")
        client.search.assert_called_once_with("test", limit=20)

    @pytest.mark.asyncio
    async def test_truncates_long_body(self) -> None:
        client = _mock_client()
        client.search.return_value = {
            "items": [
                {
                    "id": "p1",
                    "title": "Long",
                    "body": "x" * 1000,
                    "author": {"username": "u"},
                    "post_type": "discussion",
                    "score": 0,
                    "comment_count": 0,
                    "created_at": "",
                }
            ],
            "users": [],
            "total": 1,
        }
        ts = ColonyToolset(client)
        fn = ts.tools["colony_search"].function
        result = await fn(query="test")
        assert len(result["posts"][0]["body"]) == 500


class TestGetPostsTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_get_posts"].function
        result = await fn(colony="crypto", sort="top", limit=5, post_type="analysis")
        client.get_posts.assert_called_once_with(sort="top", colony="crypto", limit=5, post_type="analysis")
        assert result["posts"][0]["id"] == "post-1"
        assert result["posts"][0]["colony"] == "general"


class TestGetPostTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_get_post"].function
        result = await fn(post_id="post-1")
        client.get_post.assert_called_once_with("post-1")
        assert result["id"] == "post-1"
        assert result["body"] == "Full body text"
        assert result["author"]["username"] == "testuser"
        assert result["tags"] == ["test"]


class TestGetCommentsTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_get_comments"].function
        result = await fn(post_id="post-1", max_comments=5)
        client.iter_comments.assert_called_once_with("post-1", max_results=5)
        assert result["count"] == 1
        assert result["comments"][0]["author"] == "commenter"


class TestGetUserTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_get_user"].function
        result = await fn(user_id="user-1")
        client.get_user.assert_called_once_with("user-1")
        assert result["username"] == "testuser"
        assert result["karma"] == 42


class TestDirectoryTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_directory"].function
        result = await fn(query="python", user_type="agent", sort="newest", limit=10)
        client.directory.assert_called_once_with(query="python", user_type="agent", sort="newest", limit=10)
        assert result["users"][0]["username"] == "testuser"
        assert result["total"] == 1


class TestGetMeTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_get_me"].function
        result = await fn()
        client.get_me.assert_called_once()
        assert result["username"] == "myagent"
        assert result["karma"] == 100


class TestGetNotificationsTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_get_notifications"].function
        result = await fn(unread_only=True, limit=10)
        client.get_notifications.assert_called_once_with(unread_only=True, limit=10)
        assert result["count"] == 1
        assert result["notifications"][0]["type"] == "reply"


class TestGetPollTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_get_poll"].function
        result = await fn(post_id="post-1")
        client.get_poll.assert_called_once_with("post-1")
        assert result["total_votes"] == 10
        assert result["options"][0]["text"] == "Yes"


class TestListConversationsTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_list_conversations"].function
        result = await fn()
        client.list_conversations.assert_called_once()
        assert result["conversations"][0]["other_user"] == "otheruser"
        assert result["conversations"][0]["unread_count"] == 1


class TestGetConversationTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_get_conversation"].function
        result = await fn(username="otheruser")
        client.get_conversation.assert_called_once_with("otheruser")
        assert result["messages"][0]["sender"] == "otheruser"
        assert result["messages"][0]["body"] == "Hello!"


class TestListColoniesTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_list_colonies"].function
        result = await fn()
        client.get_colonies.assert_called_once()
        assert result["colonies"][0]["name"] == "general"


class TestCreatePostTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_create_post"].function
        result = await fn(title="New Post", body="Content", colony="findings", post_type="finding")
        client.create_post.assert_called_once_with("New Post", "Content", colony="findings", post_type="finding")
        assert result["id"] == "new-post-1"
        assert "thecolony.cc" in result["url"]


class TestCreateCommentTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_create_comment"].function
        result = await fn(post_id="post-1", body="My comment", parent_id="comment-1")
        client.create_comment.assert_called_once_with("post-1", "My comment", parent_id="comment-1")
        assert result["id"] == "new-comment-1"


class TestSendMessageTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_send_message"].function
        result = await fn(username="otheruser", body="Hi!")
        client.send_message.assert_called_once_with("otheruser", "Hi!")
        assert result["id"] == "new-msg-1"


class TestVotePostTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_vote_post"].function
        result = await fn(post_id="post-1", value=-1)
        client.vote_post.assert_called_once_with("post-1", value=-1)
        assert result["success"] is True
        assert result["vote"] == -1


class TestVoteCommentTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_vote_comment"].function
        result = await fn(comment_id="comment-1", value=1)
        client.vote_comment.assert_called_once_with("comment-1", value=1)
        assert result["success"] is True


class TestReactPostTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_react_post"].function
        result = await fn(post_id="post-1", emoji="fire")
        client.react_post.assert_called_once_with("post-1", "fire")
        assert result["emoji"] == "fire"


class TestVotePollTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_vote_poll"].function
        result = await fn(post_id="post-1", option_id="opt-1")
        client.vote_poll.assert_called_once_with("post-1", option_id="opt-1")
        assert result == {"success": True}


class TestFollowTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_follow"].function
        result = await fn(user_id="user-1")
        client.follow.assert_called_once_with("user-1")
        assert result == {"success": True}


class TestUnfollowTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_unfollow"].function
        result = await fn(user_id="user-1")
        client.unfollow.assert_called_once_with("user-1")
        assert result == {"success": True}


class TestReactCommentTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_react_comment"].function
        result = await fn(comment_id="comment-1", emoji="heart")
        client.react_comment.assert_called_once_with("comment-1", "heart")
        assert result["emoji"] == "heart"
        assert result["comment_id"] == "comment-1"


class TestUpdatePostTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_update_post"].function
        result = await fn(post_id="post-1", title="Updated Title", body="New body")
        client.update_post.assert_called_once_with("post-1", title="Updated Title", body="New body")
        assert result["title"] == "Updated Title"
        assert result["updated_at"] == "2026-01-02T00:00:00Z"


class TestDeletePostTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_delete_post"].function
        result = await fn(post_id="post-1")
        client.delete_post.assert_called_once_with("post-1")
        assert result["success"] is True


class TestMarkNotificationsReadTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_mark_notifications_read"].function
        result = await fn()
        client.mark_notifications_read.assert_called_once()
        assert result["success"] is True


class TestJoinColonyTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_join_colony"].function
        result = await fn(colony="crypto")
        client.join_colony.assert_called_once_with("crypto")
        assert result == {"success": True}


class TestLeaveColonyTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_leave_colony"].function
        result = await fn(colony="crypto")
        client.leave_colony.assert_called_once_with("crypto")
        assert result == {"success": True}


class TestGetNotificationCountTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_get_notification_count"].function
        result = await fn()
        client.get_notification_count.assert_called_once()
        assert result["count"] == 5


class TestGetUnreadCountTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_get_unread_count"].function
        result = await fn()
        client.get_unread_count.assert_called_once()
        assert result["count"] == 3


class TestIterPostsTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_iter_posts"].function
        result = await fn(colony="general", sort="top", post_type="finding", max_results=10)
        client.iter_posts.assert_called_once_with(colony="general", sort="top", post_type="finding", max_results=10)
        assert result["count"] == 1
        assert result["posts"][0]["id"] == "post-1"

    @pytest.mark.asyncio
    async def test_caps_at_200(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_iter_posts"].function
        await fn(max_results=999)
        client.iter_posts.assert_called_once_with(colony=None, sort="new", post_type=None, max_results=200)


class TestMaxBodyLength:
    @pytest.mark.asyncio
    async def test_custom_truncation(self) -> None:
        client = _mock_client()
        client.search.return_value = {
            "items": [
                {
                    "id": "p1",
                    "title": "Long",
                    "body": "x" * 1000,
                    "author": {"username": "u"},
                    "post_type": "discussion",
                    "score": 0,
                    "comment_count": 0,
                    "created_at": "",
                }
            ],
            "users": [],
            "total": 1,
        }
        ts = ColonyToolset(client, max_body_length=100)
        fn = ts.tools["colony_search"].function
        result = await fn(query="test")
        assert len(result["posts"][0]["body"]) == 100


# ── Instructions tests ───────────────────────────────────────────


class TestToolsetInstructions:
    def test_default_instructions(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        assert ts._instructions is not None

    def test_custom_instructions(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client, instructions="Custom instructions")
        assert "Custom instructions" in ts._instructions

    def test_no_instructions(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client, instructions=None)
        assert len(ts._instructions) == 0

    def test_readonly_has_instructions(self) -> None:
        client = _mock_client()
        ts = ColonyReadOnlyToolset(client)
        assert ts._instructions is not None


# ── System prompt tests ──────────────────────────────────────────


class TestColonySystemPrompt:
    @pytest.mark.asyncio
    async def test_generates_prompt(self) -> None:
        client = _mock_client()
        prompt = await colony_system_prompt(client)
        assert "@myagent" in prompt
        assert "My Agent" in prompt
        assert "100 karma" in prompt
        assert "I am an agent" in prompt
        assert "thecolony.cc" in prompt

    @pytest.mark.asyncio
    async def test_prompt_without_bio(self) -> None:
        client = _mock_client()
        client.get_me.return_value = {
            "id": "me-1",
            "username": "nobio",
            "display_name": "No Bio",
            "user_type": "agent",
            "bio": "",
            "karma": 0,
        }
        prompt = await colony_system_prompt(client)
        assert "@nobio" in prompt
        assert "Your bio:" not in prompt


# ── Error handling tests ─────────────────────────────────────────


class TestSafeResult:
    @pytest.mark.asyncio
    async def test_rate_limit_error(self) -> None:
        from colony_sdk import ColonyRateLimitError

        err = ColonyRateLimitError("Rate limited", 429, {})
        err.retry_after = 30

        @_safe_result
        async def _fn() -> dict[str, Any]:
            raise err

        result = await _fn()
        assert result["code"] == "RATE_LIMITED"
        assert result["retry_after"] == 30

    @pytest.mark.asyncio
    async def test_rate_limit_without_retry_after(self) -> None:
        from colony_sdk import ColonyRateLimitError

        err = ColonyRateLimitError("Rate limited", 429, {})

        @_safe_result
        async def _fn() -> dict[str, Any]:
            raise err

        result = await _fn()
        assert result["code"] == "RATE_LIMITED"
        assert "Please wait" in result["error"]

    @pytest.mark.asyncio
    async def test_not_found_error(self) -> None:
        from colony_sdk import ColonyNotFoundError

        @_safe_result
        async def _fn() -> dict[str, Any]:
            raise ColonyNotFoundError("Not found", 404, {})

        result = await _fn()
        assert result["code"] == "NOT_FOUND"

    @pytest.mark.asyncio
    async def test_generic_api_error(self) -> None:
        from colony_sdk import ColonyAPIError

        @_safe_result
        async def _fn() -> dict[str, Any]:
            raise ColonyAPIError("Server error", 500, {})

        result = await _fn()
        assert result["code"] == "HTTP_500"

    @pytest.mark.asyncio
    async def test_non_colony_error_propagates(self) -> None:
        @_safe_result
        async def _fn() -> dict[str, Any]:
            raise ValueError("not a colony error")

        with pytest.raises(ValueError, match="not a colony error"):
            await _fn()
