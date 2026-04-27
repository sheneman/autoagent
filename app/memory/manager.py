"""
Memory Manager
==============
File-based memory system for persistent agent context.

Stores three types of memory:
  - facts:        Key information extracted during research
  - conversations: Summaries of past research sessions
  - preferences:   User preferences and guidance

Each memory entry is a JSON file with metadata (timestamp, tags, source).
The manager supports adding, searching, and retrieving memories so agents
can build on previous work.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MemoryEntry:
    content: str
    category: str  # "facts", "conversations", "preferences"
    tags: list[str] = field(default_factory=list)
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    entry_id: str = ""


class MemoryManager:
    """Simple file-backed memory store."""

    def __init__(self, base_dir: str = "app/memory/store"):
        self.base_dir = Path(base_dir)
        for category in ("facts", "conversations", "preferences"):
            (self.base_dir / category).mkdir(parents=True, exist_ok=True)

    def add(self, entry: MemoryEntry) -> str:
        """Save a memory entry to disk. Returns the entry ID."""
        entry_id = f"{entry.category}_{int(entry.timestamp * 1000)}"
        entry.entry_id = entry_id
        path = self.base_dir / entry.category / f"{entry_id}.json"
        path.write_text(json.dumps(self._to_dict(entry), indent=2))
        return entry_id

    def search(self, query: str, category: str | None = None, limit: int = 10) -> list[MemoryEntry]:
        """Search memories by keyword match in content and tags."""
        query_lower = query.lower()
        results: list[tuple[int, MemoryEntry]] = []
        categories = [category] if category else ["facts", "conversations", "preferences"]

        for cat in categories:
            cat_dir = self.base_dir / cat
            if not cat_dir.exists():
                continue
            for fpath in cat_dir.glob("*.json"):
                try:
                    data = json.loads(fpath.read_text())
                    entry = self._from_dict(data)
                    score = 0
                    if query_lower in entry.content.lower():
                        score += 2
                    for tag in entry.tags:
                        if query_lower in tag.lower():
                            score += 1
                    if score > 0:
                        results.append((score, entry))
                except (json.JSONDecodeError, KeyError):
                    continue

        results.sort(key=lambda x: (-x[0], -x[1].timestamp))
        return [entry for _, entry in results[:limit]]

    def get_recent(self, category: str, limit: int = 5) -> list[MemoryEntry]:
        """Get the most recent entries in a category."""
        cat_dir = self.base_dir / category
        if not cat_dir.exists():
            return []

        entries: list[MemoryEntry] = []
        for fpath in cat_dir.glob("*.json"):
            try:
                data = json.loads(fpath.read_text())
                entries.append(self._from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue

        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]

    def get_all(self, category: str) -> list[MemoryEntry]:
        """Get all entries in a category."""
        return self.get_recent(category, limit=10000)

    def clear(self, category: str | None = None):
        """Remove all memories in a category, or all categories."""
        categories = [category] if category else ["facts", "conversations", "preferences"]
        for cat in categories:
            cat_dir = self.base_dir / cat
            if cat_dir.exists():
                for fpath in cat_dir.glob("*.json"):
                    fpath.unlink()

    def _to_dict(self, entry: MemoryEntry) -> dict[str, Any]:
        return {
            "entry_id": entry.entry_id,
            "content": entry.content,
            "category": entry.category,
            "tags": entry.tags,
            "source": entry.source,
            "timestamp": entry.timestamp,
        }

    def _from_dict(self, data: dict[str, Any]) -> MemoryEntry:
        return MemoryEntry(
            entry_id=data.get("entry_id", ""),
            content=data["content"],
            category=data["category"],
            tags=data.get("tags", []),
            source=data.get("source", ""),
            timestamp=data.get("timestamp", 0),
        )
