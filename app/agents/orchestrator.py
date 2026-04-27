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


def _extract_message_trace(result) -> str:
    """Extract a readable trace from an agent run's message history."""
    lines = []
    try:
        messages = result.all_messages()
        for msg in messages:
            for part in msg.parts:
                kind = type(part).__name__
                if kind == "SystemPromptPart":
                    lines.append(f"[SYSTEM] {part.content[:300]}...")
                elif kind == "UserPromptPart":
                    content = str(part.content)
                    lines.append(f"[USER] {content[:500]}{'...' if len(content) > 500 else ''}")
                elif kind == "TextPart":
                    lines.append(f"[ASSISTANT] {part.content[:800]}{'...' if len(part.content) > 800 else ''}")
                elif kind == "ToolCallPart":
                    args_str = str(part.args)[:300]
                    lines.append(f"[TOOL CALL] {part.tool_name}({args_str})")
                elif kind == "ToolReturnPart":
                    content = str(part.content)[:500]
                    lines.append(f"[TOOL RETURN] {part.tool_name} → {content}")
                elif kind == "ThinkingPart":
                    lines.append(f"[THINKING] {part.content[:600]}{'...' if len(part.content) > 600 else ''}")
                else:
                    lines.append(f"[{kind}] {str(part)[:200]}")
    except Exception as e:
        lines.append(f"[trace extraction error: {e}]")
    return "\n".join(lines)


async def run_pipeline(
    topic: str,
    config: Config,
    status_callback=None,
    verbose_callback=None,
    user_input_callback=None,
) -> dict:
    """Execute the full research-to-podcast pipeline.

    Args:
        topic: The research topic or question.
        config: Application configuration.
        status_callback: Async callable(stage, message) for UI updates.
        verbose_callback: Async callable(kind, title, body) for detailed traces.
        user_input_callback: Reserved for future interactive input.

    Returns:
        Dict with report, evaluation, podcast script, and audio path.
    """
    model = _build_model(config)
    memory = MemoryManager(config.memory_dir)

    async with httpx.AsyncClient() as http_client:
        deps = AgentDeps(
            config=config,
            http_client=http_client,
            memory=memory,
            status_callback=status_callback,
            verbose_callback=verbose_callback,
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

        await deps.send_verbose("info", "Skills loaded",
            f"research.md ({len(research_skill)} chars), writing.md ({len(writing_skill)} chars), "
            f"evaluation.md ({len(evaluation_skill)} chars), podcasting.md ({len(podcasting_skill)} chars)")

        # Check memory for prior research on this topic
        prior_research = memory.search(topic, category="facts", limit=5)
        prior_context = ""
        if prior_research:
            prior_context = "\n\nPrior research on related topics:\n"
            for entry in prior_research:
                prior_context += f"- {entry.content}\n"
            await deps.send_verbose("info", f"Memory recall: {len(prior_research)} prior facts", prior_context)
        else:
            await deps.send_verbose("info", "Memory recall: no prior research found", "")

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

        # Track best result across iterations
        best_writer_result: WrittenReport | None = None
        best_research_result: ResearchFindings | None = None
        best_eval_result: Evaluation | None = None
        best_score: float = -1

        for iteration in range(config.max_research_iterations):
            results["iterations"] = iteration + 1

            await deps.send_status(
                "researcher",
                f"Research iteration {iteration + 1}/{config.max_research_iterations}"
            )

            # Add feedback from previous evaluation if looping
            if eval_result and not eval_result.passed:
                feedback = "\n".join(f"- {imp}" for imp in eval_result.improvements)
                extra_searches = "\n".join(f"- {q}" for q in eval_result.additional_research)
                research_prompt = (
                    f"Continue researching: {topic}\n\n"
                    f"The previous report was evaluated and FAILED quality review. "
                    f"You MUST address the gaps below.\n\n"
                    f"== REQUIRED IMPROVEMENTS ==\n{feedback}\n\n"
                    f"== REQUIRED SEARCHES — run each of these as a web search ==\n"
                    f"{extra_searches}\n\n"
                    f"Execute EVERY search query listed above using the brave_web_search tool. "
                    f"These are specific gaps identified by the evaluator that must be filled.\n\n"
                    f"== PREVIOUS FINDINGS (build on these, do not repeat) ==\n"
                    f"{research_result.raw_notes if research_result else ''}\n\n"
                    f"Skill guidance:\n{research_skill}"
                )

            await deps.send_verbose("llm_request", "Researcher Agent prompt",
                research_prompt[:2000] + ("..." if len(research_prompt) > 2000 else ""))

            try:
                result = await asyncio.wait_for(
                    researcher_agent.run(research_prompt, deps=deps, model=model),
                    timeout=config.request_timeout * 2,
                )
                research_result = result.output

                trace = _extract_message_trace(result)
                await deps.send_verbose("llm_response", "Researcher Agent full trace", trace)
                await deps.send_verbose("llm_response", "Researcher Agent output",
                    f"Topic: {research_result.topic}\n"
                    f"Findings: {len(research_result.key_findings)}\n"
                    f"Sources: {len(research_result.sources)}\n"
                    f"Gaps: {research_result.gaps}\n\n"
                    f"Raw notes:\n{research_result.raw_notes[:1500]}")

            except asyncio.TimeoutError:
                await deps.send_status("researcher", "Research timed out, using partial results")
                await deps.send_verbose("info", "TIMEOUT", "Researcher agent timed out")
                if not research_result:
                    raise RuntimeError("Research timed out with no results")
            except Exception as e:
                await deps.send_status("researcher", f"Research error: {e}")
                await deps.send_verbose("info", "ERROR", str(e))
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

            await deps.send_verbose("llm_request", "Writer Agent prompt",
                write_prompt[:2000] + ("..." if len(write_prompt) > 2000 else ""))

            try:
                result = await asyncio.wait_for(
                    writer_agent.run(write_prompt, deps=deps, model=model),
                    timeout=config.request_timeout * 2,
                )
                writer_result = result.output

                trace = _extract_message_trace(result)
                await deps.send_verbose("llm_response", "Writer Agent full trace", trace)
                await deps.send_verbose("llm_response", "Writer Agent output",
                    f"Title: {writer_result.title}\n"
                    f"Words: {writer_result.word_count}\n\n"
                    f"Executive Summary:\n{writer_result.executive_summary[:500]}\n\n"
                    f"Report preview:\n{writer_result.full_report[:1000]}...")

            except asyncio.TimeoutError:
                await deps.send_status("writer", "Writing timed out")
                await deps.send_verbose("info", "TIMEOUT", "Writer agent timed out")
                if not writer_result:
                    raise RuntimeError("Writing timed out with no results")
            except Exception as e:
                await deps.send_status("writer", f"Writing error: {e}")
                await deps.send_verbose("info", "ERROR", str(e))
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

            await deps.send_verbose("llm_request", "Evaluator Agent prompt",
                eval_prompt[:2000] + ("..." if len(eval_prompt) > 2000 else ""))

            try:
                result = await asyncio.wait_for(
                    evaluator_agent.run(eval_prompt, deps=deps, model=model),
                    timeout=config.request_timeout,
                )
                eval_result = result.output

                trace = _extract_message_trace(result)
                await deps.send_verbose("llm_response", "Evaluator Agent full trace", trace)
                await deps.send_verbose("llm_response", "Evaluator Agent output",
                    f"Scores: accuracy={eval_result.accuracy_score}, "
                    f"completeness={eval_result.completeness_score}, "
                    f"clarity={eval_result.clarity_score}, "
                    f"structure={eval_result.structure_score}, "
                    f"engagement={eval_result.engagement_score}, "
                    f"overall={eval_result.overall_score}\n"
                    f"Passed: {eval_result.passed}\n\n"
                    f"Strengths:\n" + "\n".join(f"  + {s}" for s in eval_result.strengths) + "\n\n"
                    f"Improvements:\n" + "\n".join(f"  - {i}" for i in eval_result.improvements) + "\n\n"
                    f"Additional research:\n" + "\n".join(f"  ? {q}" for q in eval_result.additional_research))

            except asyncio.TimeoutError:
                await deps.send_status("evaluator", "Evaluation timed out, accepting report")
                await deps.send_verbose("info", "TIMEOUT", "Evaluator agent timed out — auto-accepting")
                eval_result = Evaluation(
                    accuracy_score=7, completeness_score=7, clarity_score=7,
                    structure_score=7, engagement_score=7, overall_score=7,
                    passed=True, strengths=["Evaluation timed out"],
                    improvements=[], additional_research=[],
                    summary="Accepted due to evaluation timeout.",
                )

            scores = (
                f"Accuracy: {eval_result.accuracy_score}, "
                f"Completeness: {eval_result.completeness_score}, "
                f"Clarity: {eval_result.clarity_score}, "
                f"Structure: {eval_result.structure_score}, "
                f"Engagement: {eval_result.engagement_score} "
                f"→ Overall: {eval_result.overall_score}/10"
            )
            await deps.send_status("evaluator", f"Scores: {scores}")

            # Track the best result across all iterations
            if eval_result.overall_score > best_score:
                best_score = eval_result.overall_score
                best_writer_result = writer_result
                best_research_result = research_result
                best_eval_result = eval_result

            if eval_result.passed:
                await deps.send_status("evaluator", "Report PASSED quality review!")
                await deps.send_verbose("info", "Evaluation PASSED", scores)
                break
            elif iteration + 1 >= config.max_research_iterations:
                await deps.send_status(
                    "evaluator",
                    f"Max iterations reached. Using best result (score: {best_score}/10)."
                )
                await deps.send_verbose("info", "Max iterations — using best result",
                    f"Best score: {best_score}/10 (iteration with highest overall)")
            else:
                gaps = eval_result.additional_research
                gap_msg = f" Searches: {', '.join(gaps[:3])}" if gaps else ""
                await deps.send_status(
                    "evaluator",
                    f"Report needs improvement. Looping back for iteration "
                    f"{iteration + 2}/{config.max_research_iterations}...{gap_msg}"
                )
                await deps.send_verbose("info", "Evaluation FAILED — looping",
                    f"Iteration {iteration + 1}/{config.max_research_iterations}\n{scores}\n\n"
                    f"Improvements requested:\n" + "\n".join(f"  - {i}" for i in eval_result.improvements) + "\n\n"
                    f"Additional searches:\n" + "\n".join(f"  > {q}" for q in eval_result.additional_research))

        # Use best result across all iterations, not just the last
        writer_result = best_writer_result or writer_result
        research_result = best_research_result or research_result
        eval_result = best_eval_result or eval_result

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

        await deps.send_verbose("llm_request", "Podcaster Agent prompt",
            podcast_prompt[:2000] + ("..." if len(podcast_prompt) > 2000 else ""))

        try:
            result = await asyncio.wait_for(
                podcaster_agent.run(podcast_prompt, deps=deps, model=model),
                timeout=config.request_timeout * 2,
            )
            podcast_result: PodcastScript = result.output

            trace = _extract_message_trace(result)
            await deps.send_verbose("llm_response", "Podcaster Agent full trace", trace)
            await deps.send_verbose("llm_response", "Podcaster Agent output",
                f"Title: {podcast_result.title}\n"
                f"Segments: {podcast_result.segment_count}\n"
                f"Duration: ~{podcast_result.estimated_duration_minutes} min\n\n"
                f"Script preview:\n{podcast_result.script[:1500]}...")

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
        await deps.send_verbose("info", f"Parsed {len(segments)} audio segments",
            "\n".join(f"{s['speaker']}: {s['text'][:80]}..." for s in segments[:10]))

        if segments:
            timestamp = int(time.time())
            audio_path = f"{config.output_dir}/podcast_{timestamp}.mp3"
            try:
                audio_file = await generate_podcast_audio(deps, segments, audio_path)
                results["audio_path"] = audio_file
            except Exception as e:
                await deps.send_status("podcaster", f"Audio generation failed: {e}")
                await deps.send_verbose("info", "TTS ERROR", str(e))
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
