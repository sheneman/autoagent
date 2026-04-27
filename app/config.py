"""
Configuration for the AutoAgent system.

Loads settings from environment variables with sensible defaults.
All external service credentials are managed here.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Central configuration for all agents and services."""

    # Mindrouter settings
    mindrouter_base_url: str = field(
        default_factory=lambda: os.getenv("MINDROUTER_BASE_URL", "https://mindrouter.uidaho.edu/v1")
    )
    mindrouter_api_key: str = field(
        default_factory=lambda: os.getenv("MINDROUTER_API_KEY", "")
    )
    mindrouter_model: str = field(
        default_factory=lambda: os.getenv("MINDROUTER_MODEL", "qwen3:32b")
    )

    # Brave Search settings
    brave_api_key: str = field(
        default_factory=lambda: os.getenv("BRAVE_API_KEY", "")
    )

    # App settings
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    max_research_iterations: int = field(
        default_factory=lambda: int(os.getenv("MAX_RESEARCH_ITERATIONS", "3"))
    )
    request_timeout: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "120"))
    )

    # TTS voice mapping for podcast hosts
    podcast_voice_a: str = field(
        default_factory=lambda: os.getenv("PODCAST_VOICE_A", "bm_george")
    )
    podcast_voice_b: str = field(
        default_factory=lambda: os.getenv("PODCAST_VOICE_B", "af_sarah")
    )

    # Paths
    output_dir: str = "output"
    memory_dir: str = "app/memory/store"
    skills_dir: str = "app/skills"


config = Config()
