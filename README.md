# AutoAgent: Multi-Agent Deep Research & Podcast Generator

A tutorial project demonstrating how to build agentic AI applications in Python.
AutoAgent takes a research topic, investigates it with multiple specialized AI
agents, produces a quality-reviewed report, and generates a two-host podcast
episode — all powered by local AI inference via [Mindrouter](https://mindrouter.uidaho.edu).

---

## Table of Contents

1. [What This Project Demonstrates](#what-this-project-demonstrates)
2. [Architecture Overview](#architecture-overview)
3. [Prerequisites](#prerequisites)
4. [Setup](#setup)
5. [Tutorial: Building It Step by Step](#tutorial-building-it-step-by-step)
   - [Step 1: Project Structure](#step-1-project-structure)
   - [Step 2: Configuration](#step-2-configuration)
   - [Step 3: Building Tools](#step-3-building-tools)
   - [Step 4: Memory System](#step-4-memory-system)
   - [Step 5: Creating Agents](#step-5-creating-agents)
   - [Step 6: The Orchestrator](#step-6-the-orchestrator)
   - [Step 7: Web Interface](#step-7-web-interface)
   - [Step 8: Podcast Generation](#step-8-podcast-generation)
6. [Running the Application](#running-the-application)
7. [How It Works](#how-it-works)
8. [Extending the System](#extending-the-system)
9. [Troubleshooting](#troubleshooting)

---

## What This Project Demonstrates

- **Multi-agent orchestration** with PydanticAI — how to build specialized agents
  that collaborate on complex tasks
- **Tool integration** — wrapping external APIs (Brave Search) as agent tools
- **Iterative evaluation loops** — using an evaluator agent as a quality gate
  with feedback-driven improvement
- **Structured outputs** — using Pydantic models to ensure agents return
  well-typed, parseable results
- **Memory management** — persistent file-based memory so agents can build on
  prior work
- **Skill files** — separating agent behavior instructions from code
- **Real-time status updates** — using Server-Sent Events (SSE) to stream
  agent activity to a web UI
- **Text-to-Speech podcast generation** — synthesizing multi-voice audio with
  Mindrouter's Kokoro TTS engine
- **Timeouts and retries** — production-grade error handling for LLM calls

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                      Web Browser                        │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Topic Input  │  │ Status Panel │  │ Results Tabs  │  │
│  └──────┬──────┘  └──────▲───────┘  └───────▲───────┘  │
└─────────┼────────────────┼──────────────────┼───────────┘
          │ POST           │ SSE              │ GET
          ▼                │                  │
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Server                         │
│                                                         │
│  ┌─────────────────── Orchestrator ──────────────────┐  │
│  │                                                   │  │
│  │  ┌────────────┐     ┌────────┐     ┌───────────┐ │  │
│  │  │ Researcher │────▶│ Writer │────▶│ Evaluator │ │  │
│  │  │            │     │        │     │           │ │  │
│  │  │ • Search   │     │        │     │ Pass? ────┼─┼──┼──▶ Podcaster ──▶ TTS Audio
│  │  │ • Scrape   │     │        │     │   │       │ │  │
│  │  │ • Memory   │     │        │     │   │ No    │ │  │
│  │  └────────────┘     └────────┘     │   ▼       │ │  │
│  │       ▲                            │ Loop back │ │  │
│  │       └────────────────────────────┘───────────┘ │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────────┐   │
│  │  Memory  │  │  Skills  │  │    Mindrouter API   │   │
│  │  (JSON)  │  │   (.md)  │  │  (LLM + TTS + STT) │   │
│  └──────────┘  └──────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Agents

| Agent | Role | Tools | Output |
|-------|------|-------|--------|
| **Researcher** | Gathers information via web search and scraping | `brave_web_search`, `scrape_url`, `recall_memory`, `save_fact` | `ResearchFindings` |
| **Writer** | Transforms research into a polished report | None (pure generation) | `WrittenReport` |
| **Evaluator** | Scores quality, provides improvement feedback | None (pure evaluation) | `Evaluation` |
| **Podcaster** | Creates a two-host podcast script | None (pure generation) | `PodcastScript` |

---

## Prerequisites

- **Python 3.11+**
- **Mindrouter API key** — Get one at https://mindrouter.uidaho.edu/dashboard
- **Brave Search API key** — Free tier at https://api-dashboard.search.brave.com
- **ffmpeg** — Required by pydub for audio processing (`brew install ffmpeg` on macOS)

---

## Setup

```bash
# Clone the repository
git clone https://github.com/sheneman/autoagent.git
cd autoagent

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your Mindrouter and Brave Search API keys
```

---

## Tutorial: Building It Step by Step

This section walks through how each component was built and why.

### Step 1: Project Structure

```
autoagent/
├── guide.md                    # System-level instructions
├── app/
│   ├── config.py               # Environment-based configuration
│   ├── main.py                 # FastAPI web application
│   ├── agents/
│   │   ├── deps.py             # Shared dependency container
│   │   ├── orchestrator.py     # Pipeline coordinator
│   │   ├── researcher.py       # Web research agent
│   │   ├── writer.py           # Report writing agent
│   │   ├── evaluator.py        # Quality evaluation agent
│   │   └── podcaster.py        # Podcast generation agent
│   ├── tools/
│   │   ├── brave_search.py     # Brave Search API wrapper
│   │   └── web_scraper.py      # HTML content extractor
│   ├── memory/
│   │   └── manager.py          # File-based memory store
│   ├── skills/                 # Agent behavior guides (markdown)
│   │   ├── research.md
│   │   ├── writing.md
│   │   ├── evaluation.md
│   │   └── podcasting.md
│   └── static/
│       └── index.html          # Web interface
└── output/                     # Generated reports and audio
```

**Why this structure?** Agents, tools, and skills are separated so each can
be modified independently. Skills are markdown files so non-developers can
tune agent behavior without touching Python code.

### Step 2: Configuration

The `Config` dataclass in `app/config.py` loads all settings from environment
variables with sensible defaults:

```python
@dataclass
class Config:
    mindrouter_base_url: str = field(
        default_factory=lambda: os.getenv("MINDROUTER_BASE_URL", "https://mindrouter.uidaho.edu/v1")
    )
    mindrouter_api_key: str = field(
        default_factory=lambda: os.getenv("MINDROUTER_API_KEY", "")
    )
    # ... more settings
```

**Key design choice:** Using a dataclass instead of raw `os.getenv()` calls
gives you type safety and a single place to see all configuration. Every
component receives the same `Config` instance through dependency injection.

### Step 3: Building Tools

Tools are async Python functions that agents can call. PydanticAI injects
dependencies via `RunContext[AgentDeps]`:

```python
@agent.tool
async def brave_web_search(ctx: RunContext[AgentDeps], query: str) -> str:
    """Search the web using Brave Search."""
    headers = {"X-Subscription-Token": ctx.deps.config.brave_api_key}
    resp = await ctx.deps.http_client.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers=headers,
        params={"q": query, "count": 8},
    )
    # ... format and return results
```

**Why httpx instead of a library?** Direct HTTP calls give full control over
timeouts, retries, and response parsing. The tool is just a function — no
framework-specific abstractions to learn.

### Step 4: Memory System

The `MemoryManager` stores facts, conversations, and preferences as JSON files:

```python
# Saving a fact during research
memory.add(MemoryEntry(
    content="GPT-4 was released in March 2023",
    category="facts",
    tags=["gpt-4", "openai", "timeline"],
    source="https://example.com/article",
))

# Recalling facts later
results = memory.search("GPT-4", category="facts")
```

**Why file-based?** It's the simplest approach that works — no database setup,
easy to inspect, and trivial to back up. For production, swap in SQLite,
Redis, or a vector database.

### Step 5: Creating Agents

Each agent is a PydanticAI `Agent` with a system prompt, dependency type,
and structured result type:

```python
class ResearchFindings(BaseModel):
    topic: str
    key_findings: list[str]
    sources: list[str]
    gaps: list[str]
    raw_notes: str

researcher_agent = Agent(
    deps_type=AgentDeps,
    output_type=ResearchFindings,
    system_prompt="You are a meticulous research agent...",
    retries=2,
)
```

**Why structured outputs?** Pydantic models ensure the agent returns data
in a predictable format. The writer can reliably access
`research.key_findings` instead of parsing free text. If the model returns
malformed data, PydanticAI retries automatically.

### Step 6: The Orchestrator

The orchestrator is plain Python — no LLM calls, just sequential logic with
an evaluation loop:

```python
for iteration in range(config.max_research_iterations):
    research = await researcher_agent.run(prompt, deps=deps, model=model)
    report = await writer_agent.run(write_prompt, deps=deps, model=model)
    evaluation = await evaluator_agent.run(eval_prompt, deps=deps, model=model)

    if evaluation.output.passed:
        break  # Quality threshold met
    # Otherwise, loop back with evaluator feedback
```

**Why not an LLM orchestrator?** An explicit Python loop is predictable,
debuggable, and testable. LLM orchestrators add latency and can make
surprising routing decisions. Use LLM orchestration when the workflow is
genuinely dynamic; use code orchestration when the steps are known.

### Step 7: Web Interface

The frontend uses vanilla HTML/CSS/JS with Server-Sent Events for real-time
status updates:

```javascript
const eventSource = new EventSource(`/api/status/${runId}`);
eventSource.onmessage = (e) => {
    const data = JSON.parse(e.data);
    addStatus(data.stage, data.message);
    updatePipeline(data.stage);  // Highlight active agent
};
```

**Why SSE instead of WebSockets?** SSE is simpler for one-way server→client
streaming. The UI only needs to receive status updates, not send frequent
messages back. One less protocol to debug.

### Step 8: Podcast Generation

The podcaster agent writes a two-host script, which is then synthesized
segment-by-segment using Mindrouter's Kokoro TTS:

```python
# Each script line is tagged with a speaker
segments = parse_script_segments(script)
# [{"speaker": "ALEX", "text": "..."}, {"speaker": "SAM", "text": "..."}, ...]

# Different voices for each host
voice_map = {"ALEX": "bm_george", "SAM": "af_heart"}

for seg in segments:
    resp = await http_client.post(
        f"{base_url}/audio/speech",
        json={"model": "kokoro", "input": seg["text"], "voice": voice_map[seg["speaker"]], "speed": 1.2},
    )
    # Append audio segment
```

Audio segments are concatenated with 400ms pauses between speakers using pydub.

---

## Running the Application

```bash
# Start the server
uvicorn app.main:app --reload --port 8000

# Open in your browser
open http://localhost:8000
```

1. Enter a research topic (e.g., "The impact of quantum computing on cryptography")
2. Watch the pipeline tracker as each agent works
3. View the generated report, evaluation scores, and podcast

---

## How It Works

1. **You enter a topic** → POST to `/api/research`
2. **Orchestrator loads skill files** and checks memory for prior research
3. **Researcher searches the web** via Brave Search, scrapes promising pages,
   stores key facts in memory
4. **Writer creates a report** using the structured research findings
5. **Evaluator scores the report** on 5 dimensions. If any score < 7 or
   overall < 7.5, the loop repeats with specific feedback
6. **Podcaster writes a script** with two hosts (Alex and Sam) discussing
   the research
7. **TTS generates audio** segment by segment using Mindrouter's Kokoro
   engine, with different voices per host
8. **Results appear in the UI** — report, evaluation scores, podcast script,
   and audio player

---

## Extending the System

### Add a new agent
1. Create `app/agents/your_agent.py` with a PydanticAI `Agent`
2. Define a Pydantic `BaseModel` for its structured output
3. Add it to the orchestrator pipeline

### Add a new tool
1. Create a function in `app/tools/` following the `RunContext[AgentDeps]` pattern
2. Register it with `agent.tool(your_function)`

### Add a new skill
1. Create `app/skills/your_skill.md` with guidelines
2. Load it in the orchestrator and inject into the agent prompt

### Use a different model
Change `MINDROUTER_MODEL` in `.env`. Good options on Mindrouter:
- `qwen3:32b` — Fast, good tool calling
- `llama3.3:70b` — Higher quality, slower
- `mistral-large` — Good balance
- `gemma-4` — Google's latest

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Missing API key" | Check your `.env` file has valid `MINDROUTER_API_KEY` and `BRAVE_API_KEY` |
| Timeouts | Increase `REQUEST_TIMEOUT` in `.env` (default: 120s) |
| TTS audio missing | Install ffmpeg: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux) |
| Agent loops forever | `MAX_RESEARCH_ITERATIONS` caps the loop (default: 3) |
| Model errors | Check available models at https://mindrouter.uidaho.edu/models |
| Port in use | Change `PORT` in `.env` or use `uvicorn app.main:app --port 8001` |

---

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Agent Framework | [PydanticAI](https://ai.pydantic.dev/) | Type-safe, fast, great docs, Python-native |
| LLM Inference | [Mindrouter](https://mindrouter.uidaho.edu) | Local, FERPA-compliant, OpenAI-compatible |
| Web Search | [Brave Search API](https://brave.com/search/api/) | Privacy-focused, generous free tier |
| TTS Engine | Kokoro (via Mindrouter) | High-quality, multiple voices |
| Web Framework | [FastAPI](https://fastapi.tiangolo.com/) | Async, fast, auto-docs |
| Status Streaming | Server-Sent Events | Simple one-way streaming |
| Audio Processing | [pydub](https://github.com/jiaaro/pydub) | Easy audio concatenation |

---

*Built as a tutorial for building agentic AI applications.*
