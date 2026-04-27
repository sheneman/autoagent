# AutoAgent System Guide

## Overview

AutoAgent is a multi-agent deep research and podcast generation system. It
demonstrates how to build production-quality agentic AI applications using
PydanticAI, Brave Search, and Mindrouter (University of Idaho's local AI
inference platform).

## Architecture

```
User → Web UI → Orchestrator
                    ├── Researcher Agent  (Brave Search + Web Scraping)
                    ├── Writer Agent      (Report Generation)
                    ├── Evaluator Agent   (Quality Assessment)
                    └── Podcaster Agent   (Script + TTS Audio)
```

## Agent Descriptions

### Orchestrator (`app/agents/orchestrator.py`)
The central coordinator. Manages the pipeline flow, handles retries and
timeouts, and broadcasts status updates to the web UI. Not an LLM agent
itself — it's Python code that calls the other agents in sequence.

### Researcher (`app/agents/researcher.py`)
Gathers information using Brave Search and web scraping. Stores key facts
in memory. Uses structured output (ResearchFindings) to ensure organized
results that downstream agents can consume reliably.

### Writer (`app/agents/writer.py`)
Transforms raw research into a polished markdown report. Follows the
structure defined in `app/skills/writing.md`. Produces structured output
(WrittenReport) with title, executive summary, full report, and key
takeaways.

### Evaluator (`app/agents/evaluator.py`)
Quality gate. Scores the report on accuracy, completeness, clarity,
structure, and engagement. Returns a pass/fail decision with specific
feedback. If the report fails, the orchestrator loops back to the
researcher and writer with the evaluator's feedback.

### Podcaster (`app/agents/podcaster.py`)
Generates a two-host podcast script and synthesizes audio using
Mindrouter's Kokoro TTS engine. The two hosts (Alex and Sam) discuss the
research in a conversational, witty style inspired by NotebookLM.

## Key Concepts

### Skill Files (`app/skills/`)
Markdown files that provide domain-specific guidance to agents. Loaded by
the orchestrator and injected into agent prompts. This separates behavioral
instructions from code, making it easy to tune agent behavior without
changing Python.

### Memory System (`app/memory/`)
File-based persistent storage for facts, conversation history, and user
preferences. Agents can recall prior research to build on past work and
avoid redundant searches.

### Evaluation Loop
The orchestrator runs a research→write→evaluate loop up to
`MAX_RESEARCH_ITERATIONS` times. Each iteration incorporates the
evaluator's feedback, progressively improving the report until it meets
quality standards or the iteration limit is reached. The orchestrator
tracks the best-scoring result across all iterations — if a later
iteration regresses, the highest-quality output is used for podcast
generation.

### Verbose Trace System
The web UI includes a toggleable verbose trace panel that displays all
agent interactions in detail: tool calls, tool results, LLM requests,
LLM responses, and informational events. This is implemented via a
`verbose_callback` passed through the dependency container (`AgentDeps`)
and streamed to the browser as SSE events with `stage="verbose"`.

### Timeouts and Retries
- Each agent call has an `asyncio.wait_for` timeout (2x `REQUEST_TIMEOUT`)
- The TTS audio generation retries up to 3 times with exponential backoff
- PydanticAI agents have `retries=2` for handling transient model errors

## Configuration

All settings are in environment variables (see `.env.example`):
- `MINDROUTER_BASE_URL` — Mindrouter API endpoint
- `MINDROUTER_API_KEY` — Your mr2_ API key from the Mindrouter dashboard
- `MINDROUTER_MODEL` — Which model to use (e.g., qwen3:32b, llama3.3:70b)
- `BRAVE_API_KEY` — Brave Search API key
- `MAX_RESEARCH_ITERATIONS` — Max evaluation loop iterations (default: 3)
- `REQUEST_TIMEOUT` — Per-request timeout in seconds (default: 120)
- `PODCAST_VOICE_A` — Kokoro TTS voice for host Alex (default: bm_george)
- `PODCAST_VOICE_B` — Kokoro TTS voice for host Sam (default: af_heart)

## Running

```bash
cp .env.example .env     # Edit with your API keys
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Then open http://localhost:8000 in your browser.
