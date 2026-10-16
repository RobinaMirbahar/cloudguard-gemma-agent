"""
Core Agent Definition for the CloudGuard Gemma Agent (Parts 1–5).

Shows how to wire an open-weights Gemma model (via Ollama locally or vLLM on
Cloud Run GPUs in production) into Google's Agent Development Kit (ADK),
complete with native Python tools, session memory, and governance callbacks.
"""

from __future__ import annotations

import os
from typing import Any

from .guardrails import inspect_and_sanitize_prompt, validate_tool_invocation
from .memory import AgentSessionState, EnterpriseMemoryBank
from .tools import (
    check_gpu_quota_and_cost,
    get_service_incident,
    list_active_alerts,
    mitigate_runaway_workload,
)

CLOUDGUARD_SYSTEM_INSTRUCTION = """You are **CloudGuard**, an Enterprise SecOps & FinOps Incident Response Agent running on Google Cloud and powered by open-weights **Gemma**.

Your primary responsibilities:
1. **Incident Triage**: Retrieve real-time operational and FinOps telemetry using `get_service_incident` and `list_active_alerts`.
2. **GPU FinOps & Capacity**: Verify regional NVIDIA L4 and RTX PRO 6000 GPU availability and monthly budget burn using `check_gpu_quota_and_cost`.
3. **Safe Remediation**: When authorized, mitigate runaway staging workloads using `mitigate_runaway_workload`.
4. **Enterprise Governance Compliance**:
   - NEVER execute automated scaling or termination in `production` without explicit `sre-incident-commander` escalation.
   - Always cite the exact Incident ID, affected service name, hourly USD burn rate, and recommended runbook ID in your response.
"""


def build_adk_root_agent() -> Any:
    """Constructs the official Google ADK `LlmAgent` backed by Gemma via LiteLLM.

    When `google-adk` is installed in the environment, returns a live `LlmAgent`
    instance compatible with `adk web` and `adk run`. Otherwise returns a
    lightweight descriptor for local deterministic simulation.
    """
    backend_mode = os.getenv("GEMMA_BACKEND_MODE", "local_ollama")
    if backend_mode == "cloud_run_vllm":
        model_id = f"openai/{os.getenv('GEMMA_VLLM_MODEL_ID', 'google/gemma-3-27b-it')}"
        api_base = os.getenv("GEMMA_VLLM_ENDPOINT_URL", "https://gemma-vllm-service-xyz-uc.a.run.app/v1")
    else:
        model_id = os.getenv("GEMMA_MODEL_NAME", "ollama_chat/gemma3:27b")
        api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

    try:
        from google.adk.agents import LlmAgent
        from google.adk.models.lite_llm import LiteLlm

        gemma_model = LiteLlm(model=model_id, api_base=api_base)
        return LlmAgent(
            name="cloudguard_gemma_agent",
            model=gemma_model,
            description="Enterprise SecOps and FinOps incident triage agent powered by Gemma.",
            instruction=CLOUDGUARD_SYSTEM_INSTRUCTION,
            tools=[
                get_service_incident,
                list_active_alerts,
                check_gpu_quota_and_cost,
                mitigate_runaway_workload,
            ],
        )
    except ImportError:
        return {
            "name": "cloudguard_gemma_agent",
            "model": model_id,
            "api_base": api_base,
            "instruction": CLOUDGUARD_SYSTEM_INSTRUCTION,
            "tools": [
                get_service_incident,
                list_active_alerts,
                check_gpu_quota_and_cost,
                mitigate_runaway_workload,
            ],
        }


class CloudGuardRuntimeOrchestrator:
    """Deterministic local & CI runtime harness that executes the full ADK lifecycle:
    1. `before_model_callback` (Model Armor & PII/Secret Redaction)
    2. Session context & Long-Term Memory retrieval
    3. Gemma Function Calling & `before_tool_callback` RBAC policy enforcement
    4. Structured response synthesis & OpenTelemetry trace recording
    """

    def __init__(self, caller_role: str = "sre-tier1-operator") -> None:
        self.caller_role = caller_role
        self.memory_bank = EnterpriseMemoryBank()
        self.sessions: dict[str, AgentSessionState] = {}

    def get_or_create_session(self, session_id: str, user_id: str = "sre-engineer@example.com") -> AgentSessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = AgentSessionState(session_id=session_id, user_id=user_id)
        return self.sessions[session_id]

    def handle_turn(self, session_id: str, user_prompt: str) -> dict[str, Any]:
        """Processes a single conversational turn with full tool, memory, and guardrail telemetry."""
        session = self.get_or_create_session(session_id)
        tool_trajectory: list[dict[str, Any]] = []

        # 1. Run Before-Model Guardrail (Part 4)
        allowed, sanitized_or_reason, model_audit = inspect_and_sanitize_prompt(user_prompt)
        if not allowed:
            session.record_turn("user", user_prompt)
            session.record_turn("model", sanitized_or_reason)
            return {
                "status": "blocked_input",
                "sanitized_prompt": user_prompt,
                "tool_trajectory": tool_trajectory,
                "audit_events": [model_audit],
                "response": sanitized_or_reason,
            }

        session.record_turn("user", sanitized_or_reason)
        lowered = sanitized_or_reason.lower()
        audit_events = [model_audit]

        # 2. Check if user references an explicit incident ID or implicitly refers to session's active incident
        import re

        inc_match = re.search(r"inc-\d{4}-\d{4}", lowered)
        incident_id = inc_match.group(0).upper() if inc_match else session.active_incident_id

        # 3. Route intent to native tools with Before-Tool Guardrail checks
        if "mitigate" in lowered or "scale down" in lowered or "terminate" in lowered:
            target_inc = incident_id or "INC-2026-0891"
            inc_info = get_service_incident(target_inc)
            target_env = inc_info.get("incident", {}).get("environment", "staging")
            if "production" in lowered or target_inc == "INC-2026-0904":
                target_env = "production"
            action = "terminate" if "terminate" in lowered else "scale_down"

            tool_args = {
                "incident_id": target_inc,
                "target_environment": target_env,
                "action": action,
                "max_instances": 1,
            }
            tool_ok, block_payload, tool_audit = validate_tool_invocation(
                "mitigate_runaway_workload", tool_args, caller_role=self.caller_role
            )
            audit_events.append(tool_audit)

            if not tool_ok:
                tool_trajectory.append(
                    {"tool": "mitigate_runaway_workload", "args": tool_args, "output": block_payload}
                )
                reply = (
                    f"Governance Policy Blocked Execution (`{block_payload['policy_id']}`): "
                    f"{block_payload['reason']}"
                )
                session.record_turn("model", reply, active_incident=target_inc)
                return {
                    "status": "blocked_tool",
                    "sanitized_prompt": sanitized_or_reason,
                    "tool_trajectory": tool_trajectory,
                    "audit_events": audit_events,
                    "response": reply,
                }

            result = mitigate_runaway_workload(**tool_args)
            tool_trajectory.append({"tool": "mitigate_runaway_workload", "args": tool_args, "output": result})
            reply = (
                f"Mitigation complete for **{target_inc}** (`{result['service_name']}` in `{target_env}`). "
                f"Applied `{action}` reducing instances from {result['previous_instances']} to "
                f"{result['new_max_instances']}. Estimated savings: **${result['estimated_savings_per_hour_usd']:.2f}/hr**."
            )
            session.record_turn("model", reply, active_incident=target_inc)
            return {
                "status": "ok",
                "sanitized_prompt": sanitized_or_reason,
                "tool_trajectory": tool_trajectory,
                "audit_events": audit_events,
                "response": reply,
            }

        if "quota" in lowered or "gpu" in lowered and ("us-central1" in lowered or "europe-west4" in lowered):
            region = "europe-west4" if "europe-west4" in lowered else "us-central1"
            tool_args = {"region": region}
            _, _, tool_audit = validate_tool_invocation(
                "check_gpu_quota_and_cost", tool_args, caller_role=self.caller_role
            )
            audit_events.append(tool_audit)
            quota_res = check_gpu_quota_and_cost(region=region)
            tool_trajectory.append({"tool": "check_gpu_quota_and_cost", "args": tool_args, "output": quota_res})
            details = quota_res["details"]
            reply = (
                f"GPU FinOps Telemetry for **{region}**:\n"
                f"- **NVIDIA L4 GPUs**: {details['nvidia_l4_in_use']}/{details['nvidia_l4_limit']} in use "
                f"({quota_res['available_nvidia_l4_gpus']} available)\n"
                f"- **NVIDIA RTX PRO 6000 GPUs**: {details['nvidia_rtx_pro_6000_in_use']}/{details['nvidia_rtx_pro_6000_limit']} in use\n"
                f"- **Monthly Spend**: ${details['current_month_spend_usd']:,.2f} / ${details['monthly_budget_usd']:,.2f} "
                f"({quota_res['budget_utilization_percent']}% of budget consumed)."
            )
            session.record_turn("model", reply)
            return {
                "status": "ok",
                "sanitized_prompt": sanitized_or_reason,
                "tool_trajectory": tool_trajectory,
                "audit_events": audit_events,
                "response": reply,
            }

        if incident_id:
            tool_args = {"incident_id": incident_id}
            _, _, tool_audit = validate_tool_invocation(
                "get_service_incident", tool_args, caller_role=self.caller_role
            )
            audit_events.append(tool_audit)
            inc_res = get_service_incident(incident_id=incident_id)
            tool_trajectory.append({"tool": "get_service_incident", "args": tool_args, "output": inc_res})
            inc = inc_res["incident"]
            memories = self.memory_bank.search_historical_resolutions(inc["service_name"])
            mem_note = ""
            if memories:
                mem_note = (
                    f"\n- **Memory Bank Match (`{memories[0]['memory_id']}`)**: {memories[0]['resolution_summary']}"
                )
            reply = (
                f"Incident Summary for **{inc['incident_id']}** ({inc['severity']} - `{inc['status']}`):\n"
                f"- **Service**: `{inc['service_name']}` (`{inc['environment']}`)\n"
                f"- **Symptom**: {inc['symptom']}\n"
                f"- **FinOps Impact**: ${inc['estimated_hourly_burn_usd']:.2f}/hr across {inc['active_instances']} instances (GPU: `{inc['gpu_type']}`)\n"
                f"- **Recommended Runbook**: `{inc['recommended_runbook']}`{mem_note}"
            )
            session.record_turn("model", reply, active_incident=incident_id)
            return {
                "status": "ok",
                "sanitized_prompt": sanitized_or_reason,
                "tool_trajectory": tool_trajectory,
                "audit_events": audit_events,
                "response": reply,
            }

        # Default: list active alerts
        env_filter = "production" if "production" in lowered else ("staging" if "staging" in lowered else "all")
        tool_args = {"environment": env_filter}
        _, _, tool_audit = validate_tool_invocation("list_active_alerts", tool_args, caller_role=self.caller_role)
        audit_events.append(tool_audit)
        alerts_res = list_active_alerts(environment=env_filter)
        tool_trajectory.append({"tool": "list_active_alerts", "args": tool_args, "output": alerts_res})
        bullet_lines = [
            f"  * **{item['incident_id']}** (`{item['severity']}` | `{item['environment']}`): `{item['service_name']}` — ${item['estimated_hourly_burn_usd']:.2f}/hr"
            for item in alerts_res["incidents"]
        ]
        reply = (
            f"Found **{alerts_res['active_count']}** active incidents (filter=`{env_filter}`):\n"
            + "\n".join(bullet_lines)
        )
        session.record_turn("model", reply)
        return {
            "status": "ok",
            "sanitized_prompt": sanitized_or_reason,
            "tool_trajectory": tool_trajectory,
            "audit_events": audit_events,
            "response": reply,
        }


root_agent = build_adk_root_agent()
