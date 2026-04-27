"""
Web Scraper Tool
================
Fetches and extracts readable text content from web pages.

Uses httpx for async HTTP requests and BeautifulSoup for HTML parsing.
Strips navigation, scripts, and styles to return clean body text.
"""

from __future__ import annotations

import httpx
from bs4 import BeautifulSoup
from pydantic_ai import RunContext
from app.agents.deps import AgentDeps

MAX_CONTENT_LENGTH = 6000


async def scrape_url(ctx: RunContext[AgentDeps], url: str) -> str:
    """Fetch a web page and extract its main text content."""
    await ctx.deps.send_verbose("tool_call", "scrape_url", f"URL: {url}")

    headers = {
        "User-Agent": "AutoAgent/1.0 (research bot)",
        "Accept": "text/html",
    }

    try:
        resp = await ctx.deps.http_client.get(
            url,
            headers=headers,
            timeout=ctx.deps.config.request_timeout,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        err = f"Failed to fetch {url}: {e}"
        await ctx.deps.send_verbose("tool_result", "scrape_url ERROR", err)
        return err

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    if len(text) > MAX_CONTENT_LENGTH:
        text = text[:MAX_CONTENT_LENGTH] + "\n\n[Content truncated...]"

    output = f"Content from {url}:\n\n{text}"
    await ctx.deps.send_verbose(
        "tool_result", f"scrape_url → {len(text)} chars", output[:2000] + ("..." if len(output) > 2000 else "")
    )
    return output
