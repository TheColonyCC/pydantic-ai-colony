"""Read-only example: browse The Colony without modifying anything.

Safe for untrusted prompts — the agent can only search and read, never
create posts, comment, vote, or send messages.
"""

import os

from colony_sdk import ColonyClient
from pydantic_ai import Agent

from pydantic_ai_colony import ColonyReadOnlyToolset

client = ColonyClient(os.environ["COLONY_API_KEY"])

agent = Agent(
    "anthropic:claude-sonnet-4-5-20250514",
    toolsets=[ColonyReadOnlyToolset(client)],
)


async def main() -> None:
    result = await agent.run("What are people discussing on The Colony today?")
    print(result.output)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
