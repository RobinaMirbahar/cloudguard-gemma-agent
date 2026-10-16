"""
Enterprise Security, RBAC, and Model Armor Guardrails for Gemma Agents (Part 4).

Implements ADK `before_model_callback` and `before_tool_callback` interceptors
to enforce zero-trust input sanitization and prevent unauthorized production mutations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Regex patterns for sensitive credentials and PII that must never reach the model or logs
SENSITIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("GCP_API_KEY", re.compile(r"AIza[0-9A-Za-z\-_]{35}")),
    ("BEARER_TOKEN", re.compile(r"ya29\.[0-9A-Za-z\-_]+")),
    ("PRIVATE_KEY_HEADER", re.compile(r"-----BEGIN (?:RSA )?PRIVATE KEY-----")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
]

PROMPT_INJECTION_SIGNATURES: list[str] = [
    "ignore all previous instructions",
    "bypass governance policy",
    "override system prompt",
    "sudo rm -rf",
]


@dataclass
class GovernanceAuditLog:
    """In-memory and structured audit log collector for security verification."""

    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event_type: str, decision: str, details: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "decision": decision,
            **details,
        }
        self.events.append(entry)
        return entry


audit_log = GovernanceAuditLog()


def inspect_and_sanitize_prompt(user_prompt: str) -> tuple[bool, str, dict[str, Any]]:
    """Scans incoming prompts for prompt injection signatures and redacts secrets.

    Mirrors Google Cloud Model Armor + ADK `before_model_callback` defense-in-depth.

    Returns:
        Tuple of (is_allowed, sanitized_prompt_or_block_reason, audit_entry).
    """
    lowered = user_prompt.lower()
    for signature in PROMPT_INJECTION_SIGNATURES:
        if signature in lowered:
            entry = audit_log.record(
                event_type="BEFORE_MODEL_GUARDRAIL",
                decision="BLOCKED_PROMPT_INJECTION",
                details={"matched_signature": signature},
            )
            return (
                False,
                f"[MODEL ARMOR BLOCK] Request rejected by security policy: detected adversarial instruction pattern ('{signature}').",
                entry,
            )

    sanitized = user_prompt
    redacted_types: list[str] = []
    for label, pattern in SENSITIVE_PATTERNS:
        if pattern.search(sanitized):
            redacted_types.append(label)
            sanitized = pattern.sub(f"[REDACTED_{label}]", sanitized)

    decision = "REDACTED_AND_ALLOWED" if redacted_types else "ALLOWED"
    entry = audit_log.record(
        event_type="BEFORE_MODEL_GUARDRAIL",
        decision=decision,
        details={"redacted_credential_types": redacted_types},
    )
    return True, sanitized, entry


def validate_tool_invocation(
    tool_name: str,
    tool_args: dict[str, Any],
    caller_role: str = "sre-tier1-operator",
) -> tuple[bool, dict[str, Any] | None, dict[str, Any]]:
    """Enforces deterministic policy checks before any ADK tool executes.

    Blocks destructive or scaling mutations in 'production' unless caller holds
    'sre-incident-commander' privilege, regardless of what the LLM decides.

    Returns:
        Tuple of (is_allowed, block_response_if_blocked, audit_entry).
    """
    if tool_name == "mitigate_runaway_workload":
        target_env = str(tool_args.get("target_environment", "")).lower()
        action = str(tool_args.get("action", "")).lower()

        if target_env == "production" and caller_role != "sre-incident-commander":
            block_payload = {
                "status": "blocked_by_governance_policy",
                "policy_id": "POL-PROD-MUTATION-001",
                "reason": (
                    f"Automated '{action}' on environment='production' is prohibited for role "
                    f"'{caller_role}'. Escalate to an 'sre-incident-commander' or execute "
                    "manual runbook verification first."
                ),
            }
            entry = audit_log.record(
                event_type="BEFORE_TOOL_GUARDRAIL",
                decision="DENIED",
                details={
                    "tool_name": tool_name,
                    "environment": target_env,
                    "caller_role": caller_role,
                    "policy_id": "POL-PROD-MUTATION-001",
                },
            )
            return False, block_payload, entry

        if action == "terminate" and target_env != "sandbox":
            block_payload = {
                "status": "blocked_by_governance_policy",
                "policy_id": "POL-NO-HARD-TERMINATE-002",
                "reason": "Hard workload termination is restricted outside 'sandbox'. Use 'scale_down' instead.",
            }
            entry = audit_log.record(
                event_type="BEFORE_TOOL_GUARDRAIL",
                decision="DENIED",
                details={
                    "tool_name": tool_name,
                    "action": action,
                    "policy_id": "POL-NO-HARD-TERMINATE-002",
                },
            )
            return False, block_payload, entry

    entry = audit_log.record(
        event_type="BEFORE_TOOL_GUARDRAIL",
        decision="ALLOWED",
        details={"tool_name": tool_name, "caller_role": caller_role},
    )
    return True, None, entry
