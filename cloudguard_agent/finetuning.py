"""
Part 6: Fine-Tuning Open-Weights Gemma with LoRA on Vertex AI for Domain-Specific Tool Calling.

Provides utilities to:
1. Convert verified CloudGuard ADK tool trajectories into Gemma SFT chat-template JSONL examples.
2. Generate the Hugging Face PEFT / QLoRA configuration targeting Gemma attention & MLP projections.
3. Build the Vertex AI Custom Training Job specification (`vertex_custom_job.yaml`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SFT_OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "cloudguard_gemma_sft_train.jsonl"
)


@dataclass
class GemmaLoRAConfig:
    """Standard QLoRA / PEFT hyperparameters optimized for Gemma instruction & tool tuning."""

    base_model_id: str = "google/gemma-3-4b-it"
    adapter_name: str = "cloudguard-sre-lora-v1"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 4096


TRAINING_TRAJECTORIES: list[dict[str, Any]] = [
    {
        "system": "You are CloudGuard, an Enterprise SecOps & FinOps Incident Response Agent.",
        "user": "We have an alert on INC-2026-0891. Pull the incident telemetry and runbook.",
        "tool_call": {
            "name": "get_service_incident",
            "arguments": {"incident_id": "INC-2026-0891"},
        },
        "tool_response": {
            "status": "success",
            "incident": {
                "incident_id": "INC-2026-0891",
                "service_name": "ml-batch-indexer-dev",
                "environment": "staging",
                "severity": "P2",
                "estimated_hourly_burn_usd": 42.50,
                "recommended_runbook": "RB-FINOPS-104: Scale max-instances to 1 and revoke batch trigger token.",
            },
        },
        "assistant_final": (
            "Incident **INC-2026-0891** (`P2` in `staging`) affects `ml-batch-indexer-dev` with an "
            "estimated burn of **$42.50/hr**. Execute runbook `RB-FINOPS-104` to scale `max-instances` to 1."
        ),
    },
    {
        "system": "You are CloudGuard, an Enterprise SecOps & FinOps Incident Response Agent.",
        "user": "Do we have enough L4 GPU quota left in us-central1 for a 2-GPU deployment?",
        "tool_call": {
            "name": "check_gpu_quota_and_cost",
            "arguments": {"region": "us-central1"},
        },
        "tool_response": {
            "status": "success",
            "region": "us-central1",
            "available_nvidia_l4_gpus": 4,
            "budget_utilization_percent": 76.8,
        },
        "assistant_final": (
            "Yes. Region **us-central1** currently has **4 available NVIDIA L4 GPUs** (12/16 in use) "
            "with monthly FinOps budget utilization at **76.8%**."
        ),
    },
    {
        "system": "You are CloudGuard, an Enterprise SecOps & FinOps Incident Response Agent.",
        "user": "Scale down the runaway staging job for INC-2026-0891 to 1 instance.",
        "tool_call": {
            "name": "mitigate_runaway_workload",
            "arguments": {
                "incident_id": "INC-2026-0891",
                "target_environment": "staging",
                "action": "scale_down",
                "max_instances": 1,
            },
        },
        "tool_response": {
            "status": "success",
            "incident_id": "INC-2026-0891",
            "service_name": "ml-batch-indexer-dev",
            "new_max_instances": 1,
            "estimated_savings_per_hour_usd": 36.12,
        },
        "assistant_final": (
            "Mitigation complete for **INC-2026-0891** (`ml-batch-indexer-dev` in `staging`). "
            "Scaled `max_instances` to 1, saving **$36.12/hr**."
        ),
    },
    {
        "system": "You are CloudGuard, an Enterprise SecOps & FinOps Incident Response Agent.",
        "user": "Show me all open production incidents right now.",
        "tool_call": {
            "name": "list_active_alerts",
            "arguments": {"environment": "production"},
        },
        "tool_response": {
            "status": "success",
            "environment_filter": "production",
            "active_count": 2,
            "incidents": [
                {"incident_id": "INC-2026-0904", "severity": "P1", "service_name": "payment-gateway-prod"},
                {"incident_id": "INC-2026-0912", "severity": "P3", "service_name": "gemma-rag-worker-prod"},
            ],
        },
        "assistant_final": (
            "There are **2** active `production` incidents: **INC-2026-0904** (`P1` on `payment-gateway-prod`) "
            "and **INC-2026-0912** (`P3` on `gemma-rag-worker-prod`)."
        ),
    },
]


def format_gemma_tool_turn(example: dict[str, Any]) -> str:
    """Formats a multi-step tool-calling trajectory using Gemma's official control tokens."""
    tool_call_json = json.dumps(example["tool_call"])
    tool_resp_json = json.dumps(example["tool_response"])
    return (
        f"<bos><start_of_turn>user\n"
        f"{example['system']}\n\n"
        f"{example['user']}<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"```tool_code\n{tool_call_json}\n```<end_of_turn>\n"
        f"<start_of_turn>user\n"
        f"```tool_output\n{tool_resp_json}\n```<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{example['assistant_final']}<end_of_turn><eos>"
    )


def export_sft_jsonl(output_path: Path = SFT_OUTPUT_PATH) -> tuple[Path, int]:
    """Writes the formatted Gemma SFT JSONL dataset to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for item in TRAINING_TRAJECTORIES:
        record = {
            "text": format_gemma_tool_turn(item),
            "metadata": {
                "target_tool": item["tool_call"]["name"],
                "domain": "secops_finops_triage",
            },
        }
        lines.append(json.dumps(record))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path, len(lines)


def build_vertex_custom_job_spec(project_id: str, bucket_name: str, config: GemmaLoRAConfig) -> dict[str, Any]:
    """Builds the Vertex AI CustomJob specification for QLoRA fine-tuning on NVIDIA L4 GPUs."""
    return {
        "displayName": f"gemma-lora-sft-{config.adapter_name}",
        "jobSpec": {
            "workerPoolSpecs": [
                {
                    "machineSpec": {
                        "machineType": "g2-standard-12",
                        "acceleratorType": "NVIDIA_L4",
                        "acceleratorCount": 1,
                    },
                    "replicaCount": 1,
                    "containerSpec": {
                        "imageUri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest",
                        "args": [
                            f"--base_model={config.base_model_id}",
                            f"--train_dataset=gs://{bucket_name}/datasets/cloudguard_gemma_sft_train.jsonl",
                            f"--output_dir=gs://{bucket_name}/adapters/{config.adapter_name}",
                            f"--lora_r={config.r}",
                            f"--lora_alpha={config.lora_alpha}",
                            f"--learning_rate={config.learning_rate}",
                            f"--epochs={config.num_train_epochs}",
                        ],
                    },
                }
            ]
        },
        "peftConfig": asdict(config),
    }
