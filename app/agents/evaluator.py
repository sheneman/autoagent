"""
Evaluator Agent
===============
Reviews a written report for quality, accuracy, and completeness.

The evaluator acts as an editorial quality gate. It scores the report
on several dimensions and provides specific, actionable feedback. If
the report doesn't meet the quality threshold, the orchestrator loops
back to the researcher and/or writer with the evaluator's feedback.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.agents.deps import AgentDeps

SYSTEM_PROMPT = """\
You are a rigorous editorial evaluator. Your job is to assess a research
report for quality and provide specific, actionable feedback that will
directly guide the next research and writing iteration.

Score each dimension from 1-10:
- Accuracy: Are claims well-supported by cited sources? Are there unsourced assertions?
- Completeness: Does it cover the topic thoroughly? What angles are missing?
- Clarity: Is it well-written and easy to understand?
- Structure: Is it well-organized with logical flow?
- Engagement: Is it interesting to read?

A report passes review if ALL scores are 7 or above AND the overall
score is 7.5 or above.

When providing feedback:
- Be specific about what needs improvement — reference exact sections or claims.
- Suggest concrete improvements, not vague directions.
- Acknowledge what the report does well.

IMPORTANT — the `additional_research` field:
If any dimension scores below 7, you MUST populate `additional_research` with
specific search queries that would fill the gaps. These queries are fed directly
to a web search tool, so write them as you would type into a search engine.
Examples:
  - "recent statistics on X 2024 2025"
  - "expert opinions on Y controversy"
  - "comparison between X and Z performance benchmarks"
Write 2-5 targeted queries that address the weakest areas of the report.
Leave the list empty ONLY if the report passes all criteria.
"""


class Evaluation(BaseModel):
    """Structured output from the evaluator agent."""

    accuracy_score: float = Field(ge=1, le=10, description="Accuracy score 1-10")
    completeness_score: float = Field(ge=1, le=10, description="Completeness score 1-10")
    clarity_score: float = Field(ge=1, le=10, description="Clarity score 1-10")
    structure_score: float = Field(ge=1, le=10, description="Structure score 1-10")
    engagement_score: float = Field(ge=1, le=10, description="Engagement score 1-10")
    overall_score: float = Field(ge=1, le=10, description="Overall weighted score")
    passed: bool = Field(description="Whether the report meets quality standards")
    strengths: list[str] = Field(description="What the report does well")
    improvements: list[str] = Field(description="Specific improvements needed")
    additional_research: list[str] = Field(
        description="Specific search queries to fill research gaps (empty if none needed)"
    )
    summary: str = Field(description="Brief evaluation summary")


evaluator_agent = Agent(
    deps_type=AgentDeps,
    output_type=Evaluation,
    system_prompt=SYSTEM_PROMPT,
    retries=2,
)
