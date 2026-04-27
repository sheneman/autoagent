"""
Brave Search Tool
=================
Wraps the Brave Search API as a PydanticAI tool.

The tool sends queries to Brave's Web Search endpoint and returns
formatted results with titles, snippets, and source URLs. This gives
research agents access to current web information.

API docs: https://api.search.brave.com/app#/web/get/web-search
"""

from __future__ import annotations

import httpx
from pydantic_ai import RunContext
from app.agents.deps import AgentDeps


async def brave_web_search(ctx: RunContext[AgentDeps], query: str) -> str:
    """Search the web using Brave Search. Returns titles, snippets, and URLs."""
    headers = {
        "X-Subscription-Token": ctx.deps.config.brave_api_key,
        "Accept": "application/json",
    }
    params = {
        "q": query,
        "count": 8,
        "search_lang": "en",
    }

    try:
        resp = await ctx.deps.http_client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
            timeout=ctx.deps.config.request_timeout,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        return f"Search API error: {e.response.status_code}"
    except httpx.RequestError as e:
        return f"Search request failed: {e}"

    data = resp.json()
    results = []
    for item in data.get("web", {}).get("results", [])[:8]:
        title = item.get("title", "")
        description = item.get("description", "")
        url = item.get("url", "")
        results.append(f"**{title}**\n{description}\nSource: {url}")

    if not results:
        return "No results found for this query."

    return "\n\n---\n\n".join(results)


async def brave_news_search(ctx: RunContext[AgentDeps], query: str) -> str:
    """Search recent news using Brave Search. Returns headlines and summaries."""
    headers = {
        "X-Subscription-Token": ctx.deps.config.brave_api_key,
        "Accept": "application/json",
    }
    params = {
        "q": query,
        "count": 5,
        "search_lang": "en",
        "freshness": "pw",  # past week
    }

    try:
        resp = await ctx.deps.http_client.get(
            "https://api.search.brave.com/res/v1/news/search",
            headers=headers,
            params=params,
            timeout=ctx.deps.config.request_timeout,
        )
        resp.raise_for_status()
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        return f"News search failed: {e}"

    data = resp.json()
    results = []
    for item in data.get("results", [])[:5]:
        title = item.get("title", "")
        description = item.get("description", "")
        url = item.get("url", "")
        age = item.get("age", "")
        results.append(f"**{title}** ({age})\n{description}\nSource: {url}")

    if not results:
        return "No recent news found for this query."

    return "\n\n---\n\n".join(results)
