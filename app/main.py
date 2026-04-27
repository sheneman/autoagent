"""
AutoAgent Web Application
=========================
FastAPI server providing:
  - A web UI for interacting with the research pipeline
  - SSE (Server-Sent Events) for real-time status updates
  - REST endpoints for starting research and retrieving results
  - Static file serving for the frontend

Run with:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from app.agents.orchestrator import run_pipeline
from app.config import config

app = FastAPI(
    title="AutoAgent",
    description="Multi-agent deep research and podcast generation",
    version="1.0.0",
)

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount output directory for serving generated audio
output_dir = Path(config.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(output_dir)), name="output")

# In-memory store for active pipelines and their status streams
active_pipelines: dict[str, dict] = {}


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main web interface."""
    html_path = static_dir / "index.html"
    return HTMLResponse(html_path.read_text())


@app.post("/api/research")
async def start_research(request: Request):
    """Start a new research pipeline.

    Expects JSON: {"topic": "your research topic"}
    Returns: {"run_id": "..."}
    """
    body = await request.json()
    topic = body.get("topic", "").strip()
    if not topic:
        return JSONResponse({"error": "Topic is required"}, status_code=400)

    run_id = f"run_{int(time.time() * 1000)}"

    # Create a status queue for SSE streaming
    status_queue: asyncio.Queue = asyncio.Queue()
    active_pipelines[run_id] = {
        "topic": topic,
        "status": "starting",
        "queue": status_queue,
        "result": None,
    }

    async def status_callback(stage: str, message: str):
        await status_queue.put({"stage": stage, "message": message, "time": time.time()})

    async def verbose_callback(kind: str, title: str, body: str):
        await status_queue.put({
            "stage": "verbose",
            "kind": kind,
            "title": title,
            "body": body,
            "time": time.time(),
        })

    async def run_in_background():
        try:
            result = await run_pipeline(
                topic=topic,
                config=config,
                status_callback=status_callback,
                verbose_callback=verbose_callback,
            )
            active_pipelines[run_id]["result"] = result
            active_pipelines[run_id]["status"] = "complete"
            await status_queue.put({
                "stage": "complete",
                "message": "Pipeline finished!",
                "time": time.time(),
            })
        except Exception as e:
            active_pipelines[run_id]["status"] = "error"
            await status_queue.put({
                "stage": "error",
                "message": str(e),
                "time": time.time(),
            })
        finally:
            await status_queue.put(None)  # Signal end of stream

    asyncio.create_task(run_in_background())

    return JSONResponse({"run_id": run_id, "topic": topic})


@app.get("/api/status/{run_id}")
async def stream_status(run_id: str):
    """Stream real-time status updates via Server-Sent Events."""
    if run_id not in active_pipelines:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    queue = active_pipelines[run_id]["queue"]

    async def event_generator():
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=300)
                if event is None:
                    break
                yield {"data": json.dumps(event)}
            except asyncio.TimeoutError:
                yield {"data": json.dumps({"stage": "keepalive", "message": "still running..."})}

    return EventSourceResponse(event_generator())


@app.get("/api/result/{run_id}")
async def get_result(run_id: str):
    """Get the final result of a completed pipeline run."""
    if run_id not in active_pipelines:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    pipeline = active_pipelines[run_id]
    if pipeline["status"] not in ("complete", "error"):
        return JSONResponse({"status": pipeline["status"], "result": None})

    return JSONResponse({"status": pipeline["status"], "result": pipeline["result"]})


@app.post("/api/user-input/{run_id}")
async def user_input(run_id: str, request: Request):
    """Send user input to a pipeline that is waiting for guidance."""
    if run_id not in active_pipelines:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    body = await request.json()
    text = body.get("text", "")

    deps = active_pipelines[run_id].get("deps")
    if deps and hasattr(deps, "user_response_event"):
        deps.user_response_text = text
        deps.user_response_event.set()
        return JSONResponse({"status": "ok"})

    return JSONResponse({"status": "no pending question"})


@app.get("/api/memory")
async def list_memories():
    """List all memories across categories."""
    from app.memory.manager import MemoryManager
    mgr = MemoryManager(config.memory_dir)
    memories = {}
    for cat in ("facts", "conversations", "preferences"):
        entries = mgr.get_all(cat)
        memories[cat] = [
            {"content": e.content, "tags": e.tags, "source": e.source, "timestamp": e.timestamp}
            for e in entries
        ]
    return JSONResponse(memories)


@app.delete("/api/memory")
async def clear_memories():
    """Clear all agent memories."""
    from app.memory.manager import MemoryManager
    mgr = MemoryManager(config.memory_dir)
    mgr.clear()
    return JSONResponse({"status": "cleared"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)
