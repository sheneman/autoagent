"""
Podcaster Agent
===============
Generates a two-host podcast script from a research report, then
synthesizes audio using Mindrouter's Kokoro TTS engine.

The podcast follows the NotebookLM style: two hosts (Alex and Sam)
discuss the research in a conversational, witty, and accessible way.
The script alternates between hosts, with natural banter, questions,
and "aha" moments that make complex topics fun to listen to.
"""

from __future__ import annotations

import io
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from app.agents.deps import AgentDeps

SYSTEM_PROMPT = """\
You are a podcast script writer who creates engaging two-host shows.
Your hosts are:
  - Alex (male voice): Enthusiastic, asks great questions, good at analogies
  - Sam (female voice): Deep knowledge, witty humor, loves surprising facts

Write a ~10 minute podcast script (about 1500 words of dialogue).

Format EVERY line as:
ALEX: [dialogue]
SAM: [dialogue]

CRITICAL RULES:
- NEVER have hosts say each other's names. Not "Thanks, Alex" or "Well, Sam"
  or any variation. Zero name usage in the dialogue text. The voices are
  distinct enough that listeners always know who is speaking.
- The ALEX:/SAM: prefixes are metadata for the TTS engine, NOT spoken aloud.
  The dialogue itself should flow as natural conversation without names.

Guidelines:
- Open with a catchy hook that draws listeners in.
- Have natural conversation flow with back-and-forth exchanges.
- Include moments of genuine surprise, humor, and "wait, really?" reactions.
- Break complex ideas into digestible explanations with analogies.
- Have hosts occasionally disagree or challenge each other.
- Include a brief mid-show recap for listeners.
- End with key takeaways and a memorable closing thought.
- Keep each speaking turn to 2-4 sentences max for natural pacing.
- Use casual language — contractions, interjections, conversational tone.
- Reference specific data and sources from the report naturally.
"""


class PodcastScript(BaseModel):
    """Structured output from the podcaster agent."""

    title: str = Field(description="Catchy podcast episode title")
    script: str = Field(description="Full script with ALEX: and SAM: prefixes")
    segment_count: int = Field(description="Number of speaking segments")
    estimated_duration_minutes: float = Field(description="Estimated duration in minutes")


podcaster_agent = Agent(
    deps_type=AgentDeps,
    output_type=PodcastScript,
    system_prompt=SYSTEM_PROMPT,
    retries=2,
)


def parse_script_segments(script: str) -> list[dict]:
    """Parse a podcast script into individual speaker segments.

    Returns a list of dicts: [{"speaker": "ALEX"|"SAM", "text": "..."}]
    """
    segments = []
    current_speaker = None
    current_text = []

    for line in script.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("ALEX:"):
            if current_speaker and current_text:
                segments.append({"speaker": current_speaker, "text": " ".join(current_text)})
            current_speaker = "ALEX"
            current_text = [line[5:].strip()]
        elif line.startswith("SAM:"):
            if current_speaker and current_text:
                segments.append({"speaker": current_speaker, "text": " ".join(current_text)})
            current_speaker = "SAM"
            current_text = [line[4:].strip()]
        elif current_speaker:
            current_text.append(line)

    if current_speaker and current_text:
        segments.append({"speaker": current_speaker, "text": " ".join(current_text)})

    return segments


async def generate_podcast_audio(
    deps: AgentDeps,
    segments: list[dict],
    output_path: str,
) -> str:
    """Generate podcast audio by synthesizing each segment with Mindrouter TTS.

    Uses different Kokoro voices for each host, then concatenates the
    segments into a single MP3 file using pydub.
    """
    from pydub import AudioSegment

    voice_map = {
        "ALEX": deps.config.podcast_voice_a,
        "SAM": deps.config.podcast_voice_b,
    }

    await deps.send_status("podcaster", f"Generating audio for {len(segments)} segments...")

    combined = AudioSegment.empty()
    pause = AudioSegment.silent(duration=400)  # 400ms pause between speakers

    for i, seg in enumerate(segments):
        voice = voice_map.get(seg["speaker"], deps.config.podcast_voice_a)
        text = seg["text"]

        await deps.send_verbose(
            "info", f"TTS segment {i + 1}/{len(segments)}",
            f"Speaker: {seg['speaker']} → Voice: {voice}\nText: {text[:120]}..."
        )
        await deps.send_status(
            "podcaster",
            f"Synthesizing segment {i + 1}/{len(segments)} ({seg['speaker']} → {voice})..."
        )

        tts_payload = {
            "model": "kokoro",
            "input": text,
            "voice": voice,
            "response_format": "mp3",
            "speed": 1.0,
        }

        resp = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = await deps.http_client.post(
                    f"{deps.config.mindrouter_base_url}/audio/speech",
                    json=tts_payload,
                    headers={"Authorization": f"Bearer {deps.config.mindrouter_api_key}"},
                    timeout=60,
                )
                resp.raise_for_status()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    await deps.send_status(
                        "podcaster",
                        f"TTS retry {attempt + 1}/{max_retries} for segment {i + 1}: {e}"
                    )
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
                else:
                    await deps.send_status("podcaster", f"TTS failed for segment {i + 1}, skipping: {e}")
                    resp = None

        if resp is None or resp.status_code != 200:
            continue

        audio_data = io.BytesIO(resp.content)
        segment_audio = AudioSegment.from_mp3(audio_data)
        combined += segment_audio + pause

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(output_file), format="mp3")

    duration_min = len(combined) / 1000 / 60
    await deps.send_status(
        "podcaster",
        f"Podcast audio generated: {duration_min:.1f} minutes ({output_file})"
    )

    return str(output_file)
