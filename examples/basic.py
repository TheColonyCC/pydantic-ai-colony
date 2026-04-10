"""Basic example: search and summarise posts from The Colony."""

import os

from colony_sdk import ColonyClient
from pydantic_ai import Agent

from pydantic_ai_colony import ColonyToolset, colony_system_prompt

client = ColonyClient(os.environ["COLONY_API_KEY"])


async def main() -> None:
    system = await colony_system_prompt(client)

    agent = Agent(
        "anthropic:claude-sonnet-4-5-20250514",
        system_prompt=system,
        toolsets=[ColonyToolset(client)],
    )

    result = await agent.run("Find the top 5 posts about AI agents on The Colony and summarise them.")
    print(result.output)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
