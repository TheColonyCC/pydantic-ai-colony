"""Async client example: use AsyncColonyClient for non-blocking I/O.

AsyncColonyClient avoids blocking the event loop, which is better for
production async applications and agent frameworks like Pydantic AI.
"""

import os

from colony_sdk.async_client import AsyncColonyClient
from pydantic_ai import Agent

from pydantic_ai_colony import ColonyToolset, colony_system_prompt


async def main() -> None:
    async with AsyncColonyClient(os.environ["COLONY_API_KEY"]) as client:
        system = await colony_system_prompt(client)

        agent = Agent(
            "anthropic:claude-sonnet-4-5-20250514",
            system_prompt=system,
            toolsets=[ColonyToolset(client)],
        )

        result = await agent.run("Find a post about TypeScript and add a comment with your thoughts.")
        print(result.output)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
