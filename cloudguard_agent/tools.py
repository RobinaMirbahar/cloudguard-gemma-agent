"""
Native Python Tools for the CloudGuard Gemma Agent (Part 2).

Each tool uses strict type hints and structured docstrings so open-weights
Gemma models can reliably generate accurate JSON function calls via Google ADK.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOCAL_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "mock_firestore.json"

DEFAULT_SEED_DATA: dict[str, Any] = {
    "incidents": {
        "INC-2026-0891": {
            "incident_id": "INC-2026-0891",
            "service_name": "ml-batch-indexer-dev",
            "environment": "staging",
            "severity": "P2",
            "status": "OPEN",
            "symptom": "Runaway GPU allocation loop on Cloud Run job; burning $42/hr above baseline.",
            "owner_team": "search-ml-platform",
            "gpu_type": "nvidia-l4",
            "active_instances": 8,
            "estimated_hourly_burn_usd": 42.50,
            "recommended_runbook": "RB-FINOPS-104: Scale max-instances to 1 and revoke batch trigger token.",
        },
        "INC-2026-0904": {
            "incident_id": "INC-2026-0904",
            "service_name": "payment-gateway-prod",
            "environment": "production",
            "severity": "P1",
            "status": "INVESTIGATING",
            "symptom": "p99 latency spike (1,850ms) following connection pool exhaustion in us-central1.",
            "owner_team": "core-payments-sre",
            "gpu_type": "none",
            "active_instances": 24,
            "estimated_hourly_burn_usd": 18.00,
            "recommended_runbook": "RB-SRE-201: Drain stale Cloud SQL proxy sockets and bump max_connections.",
        },
        "INC-2026-0912": {
            "incident_id": "INC-2026-0912",
            "service_name": "gemma-rag-worker-prod",
            "environment": "production",
            "severity": "P3",
            "status": "OPEN",
            "symptom": "Cold-start latency exceeding 14s due to uncached model weights pull.",
            "owner_team": "enterprise-ai-platform",
            "gpu_type": "nvidia-l4",
            "active_instances": 2,
            "estimated_hourly_burn_usd": 4.80,
            "recommended_runbook": "RB-AI-309: Enable Cloud Storage FUSE with parallel downloads on /mnt/models.",
        },
    },
    "quotas": {
        "us-central1": {
            "region": "us-central1",
            "nvidia_l4_limit": 16,
            "nvidia_l4_in_use": 12,
            "nvidia_rtx_pro_6000_limit": 4,
            "nvidia_rtx_pro_6000_in_use": 1,
            "monthly_budget_usd": 5000.00,
            "current_month_spend_usd": 3840.00,
        },
        "europe-west4": {
            "region": "europe-west4",
            "nvidia_l4_limit": 8,
            "nvidia_l4_in_use": 2,
            "nvidia_rtx_pro_6000_limit": 2,
            "nvidia_rtx_pro_6000_in_use": 0,
            "monthly_budget_usd": 2500.00,
            "current_month_spend_usd": 910.00,
        },
    },
}


def _load_db() -> dict[str, Any]:
    """Loads state from local JSON store or initializes from default seed data."""
    if LOCAL_DB_PATH.exists():
        return json.loads(LOCAL_DB_PATH.read_text(encoding="utf-8"))
    LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_DB_PATH.write_text(json.dumps(DEFAULT_SEED_DATA, indent=2), encoding="utf-8")
    return DEFAULT_SEED_DATA


def _save_db(data: dict[str, Any]) -> None:
    """Persists updated state to the local JSON store."""
    LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_DB_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_service_incident(incident_id: str) -> dict[str, Any]:
    """Retrieves full operational and FinOps telemetry for a specific incident ID.

    Use this tool whenever a user asks about a specific incident ticket (e.g.,
    'INC-2026-0891' or 'INC-2026-0904'), its symptoms, severity, or recommended runbook.

    Args:
        incident_id: The unique incident identifier formatted as 'INC-YYYY-NNNN'.

    Returns:
        A dictionary containing the incident status, severity, affected service,
        GPU usage, hourly burn rate, and recommended remediation runbook.
    """
    use_mock = os.getenv("USE_LOCAL_MOCK_DB", "true").lower() == "true"
    project_id = (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
    normalized_id = incident_id.strip().upper()

    if not use_mock and project_id:
        try:
            from google.cloud import firestore  # pragma: no cover

            db = firestore.Client(project=project_id)
            doc = db.collection("incidents").document(normalized_id).get()
            if not doc.exists:
                return {"status": "error", "message": f"Incident {normalized_id} not found in Firestore."}
            return {"status": "success", "incident": doc.to_dict()}
        except Exception:
            pass

    db_data = _load_db()
    incident = db_data.get("incidents", {}).get(normalized_id)
    if not incident:
        return {
            "status": "error",
            "message": f"Incident '{normalized_id}' not found. Valid IDs: {list(db_data.get('incidents', {}).keys())}",
        }
    return {"status": "success", "incident": incident}


def list_active_alerts(environment: str = "all") -> dict[str, Any]:
    """Lists all active SecOps and FinOps incident alerts, optionally filtered by environment.

    Args:
        environment: Filter by deployment environment ('production', 'staging', or 'all').

    Returns:
        A dictionary containing the count and list of matching active incidents.
    """
    db_data = _load_db()
    env_filter = environment.strip().lower()
    incidents = list(db_data.get("incidents", {}).values())
    if env_filter != "all":
        incidents = [inc for inc in incidents if inc.get("environment", "").lower() == env_filter]

    return {
        "status": "success",
        "environment_filter": env_filter,
        "active_count": len(incidents),
        "incidents": incidents,
    }


def check_gpu_quota_and_cost(region: str) -> dict[str, Any]:
    """Checks current Cloud Run GPU quota utilization and monthly FinOps spend for a GCP region.

    Use this tool when evaluating whether a region has available NVIDIA L4 or
    RTX PRO 6000 GPUs or when checking monthly budget consumption.

    Args:
        region: The Google Cloud region identifier (e.g., 'us-central1', 'europe-west4').

    Returns:
        A dictionary with GPU quota limits, active usage, and budget utilization metrics.
    """
    db_data = _load_db()
    clean_region = region.strip().lower()
    quota_info = db_data.get("quotas", {}).get(clean_region)
    if not quota_info:
        return {
            "status": "error",
            "message": f"No quota telemetry found for region '{clean_region}'. Supported: us-central1, europe-west4.",
        }

    l4_free = quota_info["nvidia_l4_limit"] - quota_info["nvidia_l4_in_use"]
    budget_pct = round(
        (quota_info["current_month_spend_usd"] / quota_info["monthly_budget_usd"]) * 100, 1
    )
    return {
        "status": "success",
        "region": clean_region,
        "available_nvidia_l4_gpus": l4_free,
        "budget_utilization_percent": budget_pct,
        "details": quota_info,
    }


def mitigate_runaway_workload(
    incident_id: str,
    target_environment: str,
    action: str,
    max_instances: int = 1,
) -> dict[str, Any]:
    """Executes an automated remediation action on a runaway Cloud Run service or job.

    IMPORTANT: Governance policy restricts automated scaling or termination in
    'production' environments without human SRE approval.

    Args:
        incident_id: The incident ID triggering this mitigation (e.g., 'INC-2026-0891').
        target_environment: The environment of the workload ('staging' or 'production').
        action: The remediation action to apply ('scale_down', 'restart_pool', or 'terminate').
        max_instances: The target maximum container instance cap (default: 1).

    Returns:
        A dictionary confirming the mitigation status and updated incident state.
    """
    db_data = _load_db()
    normalized_id = incident_id.strip().upper()
    incident = db_data.get("incidents", {}).get(normalized_id)
    if not incident:
        return {"status": "error", "message": f"Cannot mitigate unknown incident {normalized_id}."}

    previous_instances = incident["active_instances"]
    incident["active_instances"] = max_instances
    incident["status"] = "MITIGATED"
    incident["last_mitigated_at"] = datetime.now(timezone.utc).isoformat()
    _save_db(db_data)

    return {
        "status": "success",
        "incident_id": normalized_id,
        "service_name": incident["service_name"],
        "environment": target_environment,
        "action_applied": action,
        "previous_instances": previous_instances,
        "new_max_instances": max_instances,
        "estimated_savings_per_hour_usd": round(incident["estimated_hourly_burn_usd"] * 0.85, 2),
    }
