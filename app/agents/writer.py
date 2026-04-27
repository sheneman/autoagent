"""
Writer Agent
============
Transforms raw research findings into a polished, well-structured report.

The writer receives research findings from the researcher agent and
produces a comprehensive markdown report suitable for reading or
further processing (e.g., podcast script generation).

No external tools — this agent works purely with the provided content.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.agents.deps import AgentDeps

SYSTEM_PROMPT = """\
You are an expert technical writer who transforms raw research into
clear, engaging, well-structured reports.

Guidelines:
- Write in a clear, accessible style suitable for a general audience.
- Use markdown formatting with headers, bullet points, and emphasis.
- Structure the report with: Executive Summary, Key Findings (with
  subsections), Analysis, and Conclusion.
- Cite sources inline using [Source](URL) format.
- Highlight surprising or counterintuitive findings.
- Keep the tone informative but engaging — not dry or academic.
- Aim for 1500-2500 words of substantive content.
- Include specific data points, quotes, and examples from the research.
- Address any gaps noted by the researcher honestly.
"""


class WrittenReport(BaseModel):
    """Structured output from the writer agent."""

    title: str = Field(description="Compelling report title")
    executive_summary: str = Field(description="2-3 paragraph executive summary")
    full_report: str = Field(description="Complete markdown report, 1500-2500 words")
    word_count: int = Field(description="Approximate word count of the full report")
    key_takeaways: list[str] = Field(description="3-5 key takeaways for readers")


writer_agent = Agent(
    "openai:placeholder",  # model set at runtime
    deps_type=AgentDeps,
    result_type=WrittenReport,
    system_prompt=SYSTEM_PROMPT,
    retries=2,
)
