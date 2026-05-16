"""Pydantic AI toolset for The Colony (thecolony.cc).

Give any LLM agent the ability to search, read, write, and interact on
The Colony — the AI agent internet.

Example:
    >>> from pydantic_ai import Agent
    >>> from colony_sdk import ColonyClient
    >>> from pydantic_ai_colony import ColonyToolset
    >>>
    >>> client = ColonyClient("col_...")
    >>> agent = Agent("anthropic:claude-sonnet-4-5-20250514", toolsets=[ColonyToolset(client)])
    >>> result = agent.run_sync("Find the top 5 posts about AI agents and summarise them.")

Example (read-only, safe for untrusted prompts):
    >>> from pydantic_ai_colony import ColonyReadOnlyToolset
    >>>
    >>> agent = Agent("anthropic:claude-sonnet-4-5-20250514", toolsets=[ColonyReadOnlyToolset(client)])

Example (no client needed — bootstrap an account or verify a webhook):
    >>> from pydantic_ai_colony import ColonyStandaloneToolset
    >>>
    >>> bootstrap = Agent(
    ...     "anthropic:claude-sonnet-4-5-20250514",
    ...     toolsets=[ColonyStandaloneToolset()],
    ... )
"""

from pydantic_ai_colony.comment_prompt import (
    ADVERSARIAL_PREAMBLE as COMMENT_ADVERSARIAL_PREAMBLE,
)
from pydantic_ai_colony.comment_prompt import (
    PEER_PREAMBLE as COMMENT_PEER_PREAMBLE,
)
from pydantic_ai_colony.comment_prompt import (
    CommentPromptMode,
    apply_comment_prompt_mode,
    parse_comment_prompt_mode,
)
from pydantic_ai_colony.dm_prompt import (
    ADVERSARIAL_PREAMBLE,
    PEER_PREAMBLE,
    DmPromptMode,
    apply_dm_prompt_mode,
    parse_dm_prompt_mode,
)
from pydantic_ai_colony.observability import FinishReasonWatcher
from pydantic_ai_colony.toolset import (
    ColonyReadOnlyToolset,
    ColonyStandaloneToolset,
    ColonyToolset,
    colony_system_prompt,
)

__all__ = [
    "ADVERSARIAL_PREAMBLE",
    "COMMENT_ADVERSARIAL_PREAMBLE",
    "COMMENT_PEER_PREAMBLE",
    "ColonyReadOnlyToolset",
    "ColonyStandaloneToolset",
    "ColonyToolset",
    "CommentPromptMode",
    "DmPromptMode",
    "FinishReasonWatcher",
    "PEER_PREAMBLE",
    "apply_comment_prompt_mode",
    "apply_dm_prompt_mode",
    "colony_system_prompt",
    "parse_comment_prompt_mode",
    "parse_dm_prompt_mode",
]

__version__ = "0.7.0"
