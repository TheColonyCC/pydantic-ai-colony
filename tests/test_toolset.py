"""Tests for pydantic_ai_colony toolsets."""

from __future__ import annotations

import hashlib
import hmac
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from colony_sdk.async_client import AsyncColonyClient

from pydantic_ai_colony import (
    ColonyReadOnlyToolset,
    ColonyStandaloneToolset,
    ColonyToolset,
    colony_system_prompt,
)
from pydantic_ai_colony.toolset import _call, _safe_result


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
    client.get_posts_by_ids.return_value = [
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
        },
        {
            "id": "post-2",
            "title": "Second Post",
            "body": "More content",
            "author": {"username": "another", "user_type": "human"},
            "post_type": "analysis",
            "colony_id": "findings",
            "score": 12,
            "comment_count": 0,
            "created_at": "2026-01-02T00:00:00Z",
        },
    ]
    client.get_users_by_ids.return_value = [
        {
            "id": "user-1",
            "username": "alice",
            "display_name": "Alice",
            "user_type": "agent",
            "bio": "An agent",
            "karma": 7,
        },
        {
            "id": "user-2",
            "username": "bob",
            "display_name": "Bob",
            "user_type": "human",
            "bio": "A human",
            "karma": 99,
        },
    ]

    for key, value in overrides.items():
        setattr(client, key, value)
    return client


# ── Toolset structure tests ──────────────────────────────────────


class TestColonyToolset:
    def test_creates_all_tools(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        assert ts.id == "colony"
        assert len(ts.tools) == 32

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
            "colony_get_posts_by_ids",
            "colony_get_comments",
            "colony_get_user",
            "colony_get_users_by_ids",
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
        assert len(ts.tools) == 17

    def test_includes_batch_tools(self) -> None:
        client = _mock_client()
        ts = ColonyReadOnlyToolset(client)
        names = set(ts.tools.keys())
        assert "colony_get_posts_by_ids" in names
        assert "colony_get_users_by_ids" in names

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


class TestGetPostsByIdsTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_get_posts_by_ids"].function
        result = await fn(post_ids=["post-1", "post-2"])
        client.get_posts_by_ids.assert_called_once_with(["post-1", "post-2"])
        assert result["count"] == 2
        ids = [p["id"] for p in result["posts"]]
        assert ids == ["post-1", "post-2"]
        assert result["posts"][0]["author"] == "testuser"
        assert result["posts"][1]["author"] == "another"

    @pytest.mark.asyncio
    async def test_empty_list(self) -> None:
        client = _mock_client(get_posts_by_ids=MagicMock(return_value=[]))
        ts = ColonyToolset(client)
        fn = ts.tools["colony_get_posts_by_ids"].function
        result = await fn(post_ids=["nope"])
        assert result == {"posts": [], "count": 0}

    @pytest.mark.asyncio
    async def test_non_list_response_falls_back(self) -> None:
        # Defensive: if the SDK ever returns an envelope dict instead of
        # a bare list, the tool should degrade gracefully rather than
        # crashing.
        client = _mock_client(get_posts_by_ids=MagicMock(return_value={"unexpected": "envelope"}))
        ts = ColonyToolset(client)
        fn = ts.tools["colony_get_posts_by_ids"].function
        result = await fn(post_ids=["p1"])
        assert result == {"posts": [], "count": 0}


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


class TestGetUsersByIdsTool:
    @pytest.mark.asyncio
    async def test_calls_sdk(self) -> None:
        client = _mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_get_users_by_ids"].function
        result = await fn(user_ids=["user-1", "user-2"])
        client.get_users_by_ids.assert_called_once_with(["user-1", "user-2"])
        assert result["count"] == 2
        usernames = [u["username"] for u in result["users"]]
        assert usernames == ["alice", "bob"]
        assert result["users"][0]["karma"] == 7
        assert result["users"][1]["karma"] == 99

    @pytest.mark.asyncio
    async def test_empty_list(self) -> None:
        client = _mock_client(get_users_by_ids=MagicMock(return_value=[]))
        ts = ColonyToolset(client)
        fn = ts.tools["colony_get_users_by_ids"].function
        result = await fn(user_ids=["nope"])
        assert result == {"users": [], "count": 0}

    @pytest.mark.asyncio
    async def test_non_list_response_falls_back(self) -> None:
        client = _mock_client(get_users_by_ids=MagicMock(return_value={"unexpected": "envelope"}))
        ts = ColonyToolset(client)
        fn = ts.tools["colony_get_users_by_ids"].function
        result = await fn(user_ids=["u1"])
        assert result == {"users": [], "count": 0}


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


# ── Coverage gaps: async-client branches and defensive fallbacks ───
#
# The MagicMock-based tests above hit every sync branch but skip the
# `isinstance(client, AsyncColonyClient)` paths in colony_get_comments
# and colony_iter_posts, plus the `_call` helper's await branch and the
# `if not isinstance(...): ... = []` defensive fallbacks in the
# notifications / conversations / colonies tools. These tests fill the
# gap so we hold 100% line coverage.


def _async_mock_client(spec_overrides: dict[str, Any] | None = None) -> MagicMock:
    """Like ``_mock_client`` but with ``spec=AsyncColonyClient`` so the
    ``isinstance(client, AsyncColonyClient)`` checks inside the tools take
    the async branch."""

    client = MagicMock(spec=AsyncColonyClient)

    async def _async_iter(items: list[dict[str, Any]]) -> Any:
        for item in items:
            yield item

    client.iter_comments.return_value = _async_iter(
        [
            {
                "id": "comment-async-1",
                "author": {"username": "asyncbot"},
                "body": "Async comment",
                "parent_id": None,
                "score": 1,
                "created_at": "2026-01-01T00:00:00Z",
            }
        ]
    )
    client.iter_posts.return_value = _async_iter(
        [
            {
                "id": "post-async-1",
                "title": "Async Post",
                "body": "Hello from async",
                "author": {"username": "asyncbot"},
                "post_type": "discussion",
                "colony_id": "general",
                "score": 2,
                "comment_count": 0,
                "created_at": "2026-01-01T00:00:00Z",
            }
        ]
    )

    if spec_overrides:
        for key, value in spec_overrides.items():
            setattr(client, key, value)
    return client


class TestCallHelper:
    """Cover the awaitable branch of the ``_call`` dispatcher."""

    @pytest.mark.asyncio
    async def test_awaits_coroutine_results(self) -> None:
        async def _coro() -> str:
            return "awaited"

        # Pass the coroutine object directly — _call should detect it
        # has __await__ and await it.
        assert await _call(_coro()) == "awaited"

    @pytest.mark.asyncio
    async def test_returns_plain_value_unchanged(self) -> None:
        # Sync branch: a plain dict round-trips unchanged.
        assert await _call({"a": 1}) == {"a": 1}


class TestAsyncClientBranches:
    """Cover the ``isinstance(client, AsyncColonyClient)`` paths."""

    @pytest.mark.asyncio
    async def test_get_comments_async_branch(self) -> None:
        client = _async_mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_get_comments"].function
        result = await fn(post_id="post-1", max_comments=5)
        client.iter_comments.assert_called_once_with("post-1", max_results=5)
        assert result["count"] == 1
        assert result["comments"][0]["author"] == "asyncbot"

    @pytest.mark.asyncio
    async def test_iter_posts_async_branch(self) -> None:
        client = _async_mock_client()
        ts = ColonyToolset(client)
        fn = ts.tools["colony_iter_posts"].function
        result = await fn(colony="general", sort="new", max_results=10)
        client.iter_posts.assert_called_once_with(colony="general", sort="new", post_type=None, max_results=10)
        assert result["count"] == 1
        assert result["posts"][0]["id"] == "post-async-1"


class TestDefensiveFallbacks:
    """Cover the ``if not isinstance(..., list): ... = []`` branches that
    protect the tools against an unexpected SDK response shape."""

    @pytest.mark.asyncio
    async def test_get_notifications_non_list(self) -> None:
        # SDK returns a value that is neither a dict with a 'notifications'
        # key nor a list — should degrade to an empty list.
        client = _mock_client(get_notifications=MagicMock(return_value="totally unexpected"))
        ts = ColonyToolset(client)
        fn = ts.tools["colony_get_notifications"].function
        result = await fn()
        assert result == {"notifications": [], "count": 0}

    @pytest.mark.asyncio
    async def test_list_conversations_non_list(self) -> None:
        client = _mock_client(list_conversations=MagicMock(return_value="totally unexpected"))
        ts = ColonyToolset(client)
        fn = ts.tools["colony_list_conversations"].function
        result = await fn()
        assert result == {"conversations": []}

    @pytest.mark.asyncio
    async def test_list_colonies_non_list(self) -> None:
        client = _mock_client(get_colonies=MagicMock(return_value="totally unexpected"))
        ts = ColonyToolset(client)
        fn = ts.tools["colony_list_colonies"].function
        result = await fn()
        assert result == {"colonies": []}


# ── Standalone toolset (no client required) ────────────────────────


class TestColonyStandaloneToolset:
    def test_creates_toolset(self) -> None:
        ts = ColonyStandaloneToolset()
        assert ts.id == "colony-standalone"
        assert set(ts.tools.keys()) == {"colony_register", "colony_verify_webhook"}

    def test_custom_id(self) -> None:
        ts = ColonyStandaloneToolset(id="my-bootstrap")
        assert ts.id == "my-bootstrap"

    def test_no_client_required(self) -> None:
        # The whole point: instantiable without any ColonyClient.
        ts = ColonyStandaloneToolset()
        assert "colony_register" in ts.tools
        assert "colony_verify_webhook" in ts.tools

    def test_disable_instructions(self) -> None:
        ts = ColonyStandaloneToolset(instructions=None)
        # Confirm we don't crash with instructions=None.
        assert ts.id == "colony-standalone"


class TestColonyRegisterTool:
    @pytest.mark.asyncio
    async def test_returns_api_key_on_success(self) -> None:
        # ColonyClient.register is a static method on the SDK class.
        # Patch it for the duration of the test.
        with patch("pydantic_ai_colony.toolset.ColonyClient.register") as register:
            register.return_value = {
                "id": "user-new-1",
                "username": "newagent",
                "display_name": "New Agent",
                "api_key": "col_freshly_minted_key",
            }
            ts = ColonyStandaloneToolset()
            fn = ts.tools["colony_register"].function
            result = await fn(
                username="newagent",
                display_name="New Agent",
                bio="A brand-new agent",
            )
            register.assert_called_once_with("newagent", "New Agent", "A brand-new agent")
            assert result["api_key"] == "col_freshly_minted_key"
            assert result["username"] == "newagent"
            assert result["id"] == "user-new-1"

    @pytest.mark.asyncio
    async def test_handles_username_taken_error(self) -> None:
        from colony_sdk import ColonyAPIError

        with patch("pydantic_ai_colony.toolset.ColonyClient.register") as register:
            register.side_effect = ColonyAPIError("Username already taken", 409, {})
            ts = ColonyStandaloneToolset()
            fn = ts.tools["colony_register"].function
            result = await fn(username="taken", display_name="Taken", bio="...")
            # _safe_result wraps API errors as a structured error dict
            # rather than raising — the LLM gets a clear failure signal
            # without crashing the run.
            assert result["code"] == "HTTP_409"
            assert "already taken" in result["error"]


class TestColonyVerifyWebhookTool:
    @pytest.mark.asyncio
    async def test_valid_signature(self) -> None:
        secret = "supersecret"
        payload = '{"event": "post.created"}'
        sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        ts = ColonyStandaloneToolset()
        fn = ts.tools["colony_verify_webhook"].function
        result = await fn(payload=payload, signature=sig, secret=secret)
        assert result == {"valid": True}

    @pytest.mark.asyncio
    async def test_invalid_signature(self) -> None:
        ts = ColonyStandaloneToolset()
        fn = ts.tools["colony_verify_webhook"].function
        result = await fn(
            payload='{"event": "post.created"}',
            signature="0" * 64,
            secret="supersecret",
        )
        assert result == {"valid": False}

    @pytest.mark.asyncio
    async def test_sha256_prefix_tolerated(self) -> None:
        # The SDK function strips a leading "sha256=" prefix for
        # framework compatibility (Stripe, GitHub, etc. use that style).
        secret = "supersecret"
        payload = "raw bytes here"
        sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        ts = ColonyStandaloneToolset()
        fn = ts.tools["colony_verify_webhook"].function
        result = await fn(
            payload=payload,
            signature=f"sha256={sig}",
            secret=secret,
        )
        assert result == {"valid": True}

    @pytest.mark.asyncio
    async def test_returns_error_dict_on_exception(self) -> None:
        # Patch the underlying function to raise so we cover the
        # exception branch. (In practice the only way to trigger this is
        # to pass garbage that the SDK can't handle, but mocking is
        # cleaner and doesn't depend on the SDK's input validation.)
        with patch("pydantic_ai_colony.toolset.verify_webhook") as vw:
            vw.side_effect = ValueError("malformed signature hex")
            ts = ColonyStandaloneToolset()
            fn = ts.tools["colony_verify_webhook"].function
            result = await fn(payload="x", signature="x", secret="x")
            assert result["valid"] is False
            assert "malformed" in result["error"]
