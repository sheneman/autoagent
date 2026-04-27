"""
Shared Dependencies
===================
Defines the dependency container that all agents receive via PydanticAI's
dependency injection. This keeps HTTP clients, config, and shared state
in one place rather than scattered across global variables.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from app.config import Config
    from app.memory.manager import MemoryManager


@dataclass
class AgentDeps:
    """Injected into every agent run via PydanticAI's RunContext."""

    config: Config
    http_client: httpx.AsyncClient
    memory: MemoryManager

    # Callback to send status updates to the web UI
    status_callback: object = None  # async callable(stage, message)

    # Event for pausing the workflow to ask the user a question
    user_response_event: asyncio.Event = field(default_factory=asyncio.Event)
    user_response_text: str = ""

    async def send_status(self, stage: str, message: str):
        """Broadcast a status update to the web frontend."""
        if self.status_callback:
            await self.status_callback(stage, message)

    async def ask_user(self, question: str) -> str:
        """Pause the workflow and ask the user for input via the web UI."""
        self.user_response_event.clear()
        await self.send_status("question", question)
        await self.user_response_event.wait()
        return self.user_response_text
