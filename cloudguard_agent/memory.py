"""
Session State and Long-Term Memory Bank for Gemma Enterprise Agents (Part 3).

Demonstrates both short-term conversational session state (active incident context,
operator preferences) and cross-session organizational memory (past postmortems
and historical resolution learnings).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AgentSessionState:
    """Short-term session state scoped to a single SRE troubleshooting thread."""

    session_id: str
    user_id: str
    active_incident_id: str | None = None
    preferred_region: str = "us-central1"
    turn_history: list[dict[str, str]] = field(default_factory=list)
    context_flags: dict[str, Any] = field(default_factory=dict)

    def record_turn(self, role: str, content: str, active_incident: str | None = None) -> None:
        if active_incident:
            self.active_incident_id = active_incident
        self.turn_history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "role": role,
                "content": content,
            }
        )


class EnterpriseMemoryBank:
    """Long-term cross-session memory store for resolved incident lessons."""

    def __init__(self) -> None:
        self._memories: list[dict[str, Any]] = [
            {
                "memory_id": "MEM-2026-014",
                "service_name": "ml-batch-indexer-dev",
                "pattern": "Runaway GPU allocation loop on Cloud Run job",
                "resolution_summary": (
                    "Root cause was an unthrottled Pub/Sub retry policy re-triggering batch workers. "
                    "Scaling max-instances to 1 immediately halted the $42/hr L4 GPU burn."
                ),
                "verified_by": "sre-oncall@example.com",
            },
            {
                "memory_id": "MEM-2026-029",
                "service_name": "gemma-rag-worker-prod",
                "pattern": "Cold-start latency exceeding 14s on Gemma 27B container boot",
                "resolution_summary": (
                    "Mounting the GCS bucket via Cloud Storage FUSE with parallel downloads "
                    "and --gpu-zonal-redundancy reduced cold start from 14.2s to 3.8s."
                ),
                "verified_by": "ai-infra-lead@example.com",
            },
        ]

    def search_historical_resolutions(self, query: str) -> list[dict[str, Any]]:
        """Searches long-term memory for similar historical incidents or services."""
        tokens = [t.lower() for t in query.split() if len(t) > 2]
        matches: list[dict[str, Any]] = []
        for mem in self._memories:
            haystack = f"{mem['service_name']} {mem['pattern']} {mem['resolution_summary']}".lower()
            if any(tok in haystack for tok in tokens):
                matches.append(mem)
        return matches

    def store_resolution_memory(
        self, service_name: str, pattern: str, resolution_summary: str, verified_by: str
    ) -> dict[str, Any]:
        """Commits a newly verified incident resolution into the long-term memory bank."""
        new_entry = {
            "memory_id": f"MEM-2026-{len(self._memories) + 35:03d}",
            "service_name": service_name,
            "pattern": pattern,
            "resolution_summary": resolution_summary,
            "verified_by": verified_by,
        }
        self._memories.append(new_entry)
        return new_entry
