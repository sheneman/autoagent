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
report for quality and provide specific, actionable feedback.

Score each dimension from 1-10:
- Accuracy: Are claims well-supported by cited sources?
- Completeness: Does it cover the topic thoroughly?
- Clarity: Is it well-written and easy to understand?
- Structure: Is it well-organized with logical flow?
- Engagement: Is it interesting to read?

A report passes review if ALL scores are 7 or above AND the overall
score is 7.5 or above.

When providing feedback:
- Be specific about what needs improvement.
- Reference exact sections or claims that need work.
- Suggest concrete improvements, not vague directions.
- If research gaps exist, specify what searches to run.
- Acknowledge what the report does well.
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
    "openai:placeholder",  # model set at runtime
    deps_type=AgentDeps,
    result_type=Evaluation,
    system_prompt=SYSTEM_PROMPT,
    retries=2,
)
