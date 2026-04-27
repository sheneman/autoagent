"""
Researcher Agent
================
Gathers information on a topic using web search and page scraping.

This agent receives a research brief from the orchestrator and performs
multiple search queries to build a comprehensive set of findings. It
stores key facts in memory for future reference.

Tools: brave_web_search, brave_news_search, scrape_url
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from app.agents.deps import AgentDeps
from app.memory.manager import MemoryEntry
from app.tools.brave_search import brave_news_search, brave_web_search
from app.tools.web_scraper import scrape_url

SYSTEM_PROMPT = """\
You are a meticulous research agent. Your job is to investigate a topic
thoroughly using web search and page scraping tools.

Guidelines:
- Start with broad searches, then drill into specific subtopics.
- Search for multiple angles: facts, opinions, recent developments, data.
- When you find a promising result, scrape the page for deeper content.
- Always note your sources with URLs.
- Organize your findings clearly with sections and bullet points.
- Aim for depth and accuracy over breadth.
- If you have prior context from memory, build on it rather than repeating.

You MUST use the search tools to find information. Do not make up facts.
"""


class ResearchFindings(BaseModel):
    """Structured output from the researcher agent."""

    topic: str = Field(description="The research topic")
    key_findings: list[str] = Field(description="Main findings, each a detailed paragraph")
    sources: list[str] = Field(description="URLs of sources used")
    subtopics_explored: list[str] = Field(description="Subtopics that were investigated")
    gaps: list[str] = Field(description="Areas that need more research")
    raw_notes: str = Field(description="Detailed raw research notes with citations")


researcher_agent = Agent(
    deps_type=AgentDeps,
    output_type=ResearchFindings,
    system_prompt=SYSTEM_PROMPT,
    retries=2,
)

researcher_agent.tool(brave_web_search)
researcher_agent.tool(brave_news_search)
researcher_agent.tool(scrape_url)


@researcher_agent.tool
async def recall_memory(ctx: RunContext[AgentDeps], query: str) -> str:
    """Search agent memory for prior research on a topic."""
    entries = ctx.deps.memory.search(query, category="facts", limit=5)
    if not entries:
        return "No prior research found on this topic."
    results = []
    for e in entries:
        results.append(f"[{e.source}] {e.content}")
    return "\n\n".join(results)


@researcher_agent.tool
async def save_fact(ctx: RunContext[AgentDeps], fact: str, tags: str, source: str) -> str:
    """Save an important fact to memory for future reference. Tags should be comma-separated."""
    entry = MemoryEntry(
        content=fact,
        category="facts",
        tags=[t.strip() for t in tags.split(",")],
        source=source,
    )
    entry_id = ctx.deps.memory.add(entry)
    return f"Fact saved (id: {entry_id})"
