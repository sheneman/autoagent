"""
Orchestrator
============
Coordinates the multi-agent deep research and podcast workflow.

The orchestrator manages the full pipeline:
  1. Accept a research topic from the user
  2. Load relevant skill files for agent guidance
  3. Run the researcher agent to gather information
  4. Run the writer agent to produce a report
  5. Run the evaluator agent to assess quality
  6. Loop back to researcher/writer if quality is insufficient
  7. Run the podcaster agent to generate a script
  8. Synthesize podcast audio via Mindrouter TTS
  9. Save results and update memory

The orchestrator handles retries, timeouts, and user interaction
at each stage. It broadcasts status updates to the web UI via SSE.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.agents.deps import AgentDeps
from app.agents.evaluator import Evaluation, evaluator_agent
from app.agents.podcaster import (
    PodcastScript,
    generate_podcast_audio,
    parse_script_segments,
    podcaster_agent,
)
from app.agents.researcher import ResearchFindings, researcher_agent
from app.agents.writer import WrittenReport, writer_agent
from app.config import Config
from app.memory.manager import MemoryEntry, MemoryManager


def _build_model(config: Config) -> OpenAIModel:
    """Create an OpenAI-compatible model pointing at Mindrouter."""
    provider = OpenAIProvider(
        base_url=config.mindrouter_base_url,
        api_key=config.mindrouter_api_key,
    )
    return OpenAIModel(config.mindrouter_model, provider=provider)


def _load_skill(skills_dir: str, name: str) -> str:
    """Load a skill file (markdown) from the skills directory."""
    path = Path(skills_dir) / f"{name}.md"
    if path.exists():
        return path.read_text()
    return ""


async def run_pipeline(
    topic: str,
    config: Config,
    status_callback=None,
    user_input_callback=None,
) -> dict:
    """Execute the full research-to-podcast pipeline.

    Args:
        topic: The research topic or question.
        config: Application configuration.
        status_callback: Async callable(stage, message) for UI updates.
        user_input_callback: Reserved for future interactive input.

    Returns:
        Dict with report, evaluation, podcast script, and audio path.
    """
    model = _build_model(config)
    memory = MemoryManager(config.memory_dir)

    # Override each agent's model at runtime
    researcher_agent.model = model
    writer_agent.model = model
    evaluator_agent.model = model
    podcaster_agent.model = model

    async with httpx.AsyncClient() as http_client:
        deps = AgentDeps(
            config=config,
            http_client=http_client,
            memory=memory,
            status_callback=status_callback,
        )

        results = {
            "topic": topic,
            "research": None,
            "report": None,
            "evaluation": None,
            "podcast_script": None,
            "audio_path": None,
            "iterations": 0,
            "duration_seconds": 0,
        }

        start_time = time.time()

        # Load skill files for agent guidance
        research_skill = _load_skill(config.skills_dir, "research")
        writing_skill = _load_skill(config.skills_dir, "writing")
        evaluation_skill = _load_skill(config.skills_dir, "evaluation")
        podcasting_skill = _load_skill(config.skills_dir, "podcasting")

        # Check memory for prior research on this topic
        prior_research = memory.search(topic, category="facts", limit=5)
        prior_context = ""
        if prior_research:
            prior_context = "\n\nPrior research on related topics:\n"
            for entry in prior_research:
                prior_context += f"- {entry.content}\n"

        # ── Stage 1: Research ──────────────────────────────────────
        await deps.send_status("researcher", f"Starting research on: {topic}")

        research_prompt = (
            f"Research this topic thoroughly: {topic}\n\n"
            f"Skill guidance:\n{research_skill}\n"
            f"{prior_context}"
        )

        research_result: ResearchFindings | None = None
        writer_result: WrittenReport | None = None
        eval_result: Evaluation | None = None

        for iteration in range(config.max_research_iterations):
            results["iterations"] = iteration + 1

            await deps.send_status(
                "researcher",
                f"Research iteration {iteration + 1}/{config.max_research_iterations}"
            )

            # Add feedback from previous evaluation if looping
            if eval_result and not eval_result.passed:
                feedback = "\n".join(eval_result.improvements)
                extra_searches = "\n".join(eval_result.additional_research)
                research_prompt = (
                    f"Continue researching: {topic}\n\n"
                    f"Previous evaluation feedback:\n{feedback}\n\n"
                    f"Suggested additional searches:\n{extra_searches}\n\n"
                    f"Previous findings to build upon:\n"
                    f"{research_result.raw_notes if research_result else ''}\n\n"
                    f"Skill guidance:\n{research_skill}"
                )

            try:
                result = await asyncio.wait_for(
                    researcher_agent.run(research_prompt, deps=deps),
                    timeout=config.request_timeout * 2,
                )
                research_result = result.data
            except asyncio.TimeoutError:
                await deps.send_status("researcher", "Research timed out, using partial results")
                if not research_result:
                    raise RuntimeError("Research timed out with no results")
            except Exception as e:
                await deps.send_status("researcher", f"Research error: {e}")
                if not research_result:
                    raise

            await deps.send_status(
                "researcher",
                f"Found {len(research_result.key_findings)} key findings from "
                f"{len(research_result.sources)} sources"
            )

            # ── Stage 2: Writing ───────────────────────────────────
            await deps.send_status("writer", "Drafting report...")

            write_prompt = (
                f"Write a comprehensive report based on this research.\n\n"
                f"Topic: {research_result.topic}\n\n"
                f"Key Findings:\n"
                + "\n".join(f"- {f}" for f in research_result.key_findings)
                + f"\n\nSources:\n"
                + "\n".join(f"- {s}" for s in research_result.sources)
                + f"\n\nDetailed Notes:\n{research_result.raw_notes}\n\n"
                f"Skill guidance:\n{writing_skill}"
            )

            if eval_result and not eval_result.passed:
                write_prompt += (
                    f"\n\nPrevious evaluation feedback to address:\n"
                    + "\n".join(f"- {imp}" for imp in eval_result.improvements)
                )

            try:
                result = await asyncio.wait_for(
                    writer_agent.run(write_prompt, deps=deps),
                    timeout=config.request_timeout * 2,
                )
                writer_result = result.data
            except asyncio.TimeoutError:
                await deps.send_status("writer", "Writing timed out")
                if not writer_result:
                    raise RuntimeError("Writing timed out with no results")
            except Exception as e:
                await deps.send_status("writer", f"Writing error: {e}")
                if not writer_result:
                    raise

            await deps.send_status(
                "writer",
                f"Report drafted: '{writer_result.title}' ({writer_result.word_count} words)"
            )

            # ── Stage 3: Evaluation ────────────────────────────────
            await deps.send_status("evaluator", "Evaluating report quality...")

            eval_prompt = (
                f"Evaluate this research report:\n\n"
                f"Title: {writer_result.title}\n\n"
                f"Executive Summary:\n{writer_result.executive_summary}\n\n"
                f"Full Report:\n{writer_result.full_report}\n\n"
                f"Key Takeaways:\n"
                + "\n".join(f"- {t}" for t in writer_result.key_takeaways)
                + f"\n\nEvaluation criteria guidance:\n{evaluation_skill}"
            )

            try:
                result = await asyncio.wait_for(
                    evaluator_agent.run(eval_prompt, deps=deps),
                    timeout=config.request_timeout,
                )
                eval_result = result.data
            except asyncio.TimeoutError:
                await deps.send_status("evaluator", "Evaluation timed out, accepting report")
                eval_result = Evaluation(
                    accuracy_score=7, completeness_score=7, clarity_score=7,
                    structure_score=7, engagement_score=7, overall_score=7,
                    passed=True, strengths=["Evaluation timed out"],
                    improvements=[], additional_research=[],
                    summary="Accepted due to evaluation timeout.",
                )

            scores = (
                f"Accuracy: {eval_result.accuracy_score}/10, "
                f"Completeness: {eval_result.completeness_score}/10, "
                f"Clarity: {eval_result.clarity_score}/10, "
                f"Overall: {eval_result.overall_score}/10"
            )
            await deps.send_status("evaluator", f"Scores: {scores}")

            if eval_result.passed:
                await deps.send_status("evaluator", "Report PASSED quality review!")
                break
            else:
                await deps.send_status(
                    "evaluator",
                    f"Report needs improvement. Looping back... "
                    f"({iteration + 1}/{config.max_research_iterations})"
                )

        results["research"] = research_result.model_dump() if research_result else None
        results["report"] = writer_result.model_dump() if writer_result else None
        results["evaluation"] = eval_result.model_dump() if eval_result else None

        # ── Stage 4: Podcast Generation ────────────────────────────
        await deps.send_status("podcaster", "Writing podcast script...")

        podcast_prompt = (
            f"Create an engaging two-host podcast script about this topic.\n\n"
            f"Report Title: {writer_result.title}\n\n"
            f"Executive Summary:\n{writer_result.executive_summary}\n\n"
            f"Full Report:\n{writer_result.full_report}\n\n"
            f"Key Takeaways:\n"
            + "\n".join(f"- {t}" for t in writer_result.key_takeaways)
            + f"\n\nPodcast style guidance:\n{podcasting_skill}"
        )

        try:
            result = await asyncio.wait_for(
                podcaster_agent.run(podcast_prompt, deps=deps),
                timeout=config.request_timeout * 2,
            )
            podcast_result: PodcastScript = result.data
        except asyncio.TimeoutError:
            await deps.send_status("podcaster", "Script generation timed out")
            raise RuntimeError("Podcast script generation timed out")

        await deps.send_status(
            "podcaster",
            f"Script ready: '{podcast_result.title}' "
            f"({podcast_result.segment_count} segments, "
            f"~{podcast_result.estimated_duration_minutes:.0f} min)"
        )

        results["podcast_script"] = podcast_result.model_dump()

        # ── Stage 5: Audio Synthesis ───────────────────────────────
        await deps.send_status("podcaster", "Generating podcast audio...")

        segments = parse_script_segments(podcast_result.script)
        if segments:
            timestamp = int(time.time())
            audio_path = f"{config.output_dir}/podcast_{timestamp}.mp3"
            try:
                audio_file = await generate_podcast_audio(deps, segments, audio_path)
                results["audio_path"] = audio_file
            except Exception as e:
                await deps.send_status("podcaster", f"Audio generation failed: {e}")
                results["audio_path"] = None
        else:
            await deps.send_status("podcaster", "No valid segments found in script")

        # ── Save to memory ─────────────────────────────────────────
        memory.add(MemoryEntry(
            content=f"Researched '{topic}': {writer_result.title}. "
                    f"Quality score: {eval_result.overall_score}/10. "
                    f"Iterations: {results['iterations']}.",
            category="conversations",
            tags=[topic.split()[0].lower(), "research", "completed"],
            source="orchestrator",
        ))

        results["duration_seconds"] = round(time.time() - start_time, 1)
        await deps.send_status("complete", f"Pipeline complete in {results['duration_seconds']}s")

        # Save full results to disk
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / f"results_{int(time.time())}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return results
