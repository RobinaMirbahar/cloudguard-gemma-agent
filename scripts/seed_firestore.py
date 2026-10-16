"""
Seeds the CloudGuard Incident & FinOps database (Cloud Firestore or local mock store).
Used in Part 2 of the tutorial series.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloudguard_agent.tools import DEFAULT_SEED_DATA, LOCAL_DB_PATH


def seed_database() -> None:
    use_mock = os.getenv("USE_LOCAL_MOCK_DB", "true").lower() == "true"
    project_id = (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip() or "cloudguard-enterprise-demo"

    print("=" * 74)
    print(f"CloudGuard Database Seeder | Project: {project_id}")

    if not use_mock and project_id == "cloudguard-enterprise-demo":
        print("[INFO] No active GCP project set in GOOGLE_CLOUD_PROJECT; using Local Firestore Schema Mirror.")
        use_mock = True

    print(f"Target Mode: {'Local JSON Store (Firestore Schema Mirror)' if use_mock else 'Cloud Firestore Native'}")
    print("=" * 74)

    if not use_mock:
        try:
            from google.cloud import firestore  # pragma: no cover

            db = firestore.Client(project=project_id)
            for doc_id, payload in DEFAULT_SEED_DATA["incidents"].items():
                db.collection("incidents").document(doc_id).set(payload)
            for doc_id, payload in DEFAULT_SEED_DATA["quotas"].items():
                db.collection("quotas").document(doc_id).set(payload)
            print("[OK] Seeded Cloud Firestore collections: 'incidents' and 'quotas'")
        except Exception as exc:
            print(f"[WARN] Cloud Firestore unavailable for project '{project_id}' ({exc.__class__.__name__}).")
            print("[INFO] Falling back to Local Firestore Schema Mirror.")
            use_mock = True

    if use_mock:
        LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_DB_PATH.write_text(json.dumps(DEFAULT_SEED_DATA, indent=2), encoding="utf-8")
        print(f"[OK] Wrote seed collections to: {LOCAL_DB_PATH}")

    print("\nSeeded Collection Summary [incidents]:")
    print(f"{'INCIDENT ID':<16} {'SEVERITY':<10} {'ENVIRONMENT':<13} {'SERVICE NAME':<26} {'BURN ($/HR)':<12}")
    print("-" * 74)
    for inc in DEFAULT_SEED_DATA["incidents"].values():
        print(
            f"{inc['incident_id']:<16} {inc['severity']:<10} {inc['environment']:<13} "
            f"{inc['service_name']:<26} ${inc['estimated_hourly_burn_usd']:<.2f}"
        )

    print("\nSeeded Collection Summary [quotas]:")
    print(f"{'REGION':<16} {'L4 IN USE / LIMIT':<22} {'RTX 6000 IN USE / LIMIT':<25} {'MONTHLY SPEND':<15}")
    print("-" * 74)
    for q in DEFAULT_SEED_DATA["quotas"].values():
        l4_str = f"{q['nvidia_l4_in_use']}/{q['nvidia_l4_limit']}"
        rtx_str = f"{q['nvidia_rtx_pro_6000_in_use']}/{q['nvidia_rtx_pro_6000_limit']}"
        spend_str = f"${q['current_month_spend_usd']:,.0f} / ${q['monthly_budget_usd']:,.0f}"
        print(f"{q['region']:<16} {l4_str:<22} {rtx_str:<25} {spend_str:<15}")
    print("=" * 74)


if __name__ == "__main__":
    seed_database()
