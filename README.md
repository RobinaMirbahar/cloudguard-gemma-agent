# CloudGuard: Open-Weights Enterprise SecOps & FinOps Agent (`Gemma` + `Google ADK` + `Cloud Run GPUs`)

[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/?cloudshell_git_repo=https://github.com/RobinaMirbahar/cloudguard-gemma-agent)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Google ADK](https://img.shields.io/badge/Google-Agent%20Development%20Kit-4285F4?logo=googlecloud&logoColor=white)](https://google.github.io/adk-docs/)
[![Model: Gemma](https://img.shields.io/badge/Open%20Weights-Gemma%203-34A853?logo=google&logoColor=white)](https://ai.google.dev/gemma)
[![Runtime: Cloud Run GPU](https://img.shields.io/badge/Cloud%20Run-Serverless%20GPU%20(vLLM)-FBBC04?logo=googlecloud&logoColor=black)](https://cloud.google.com/run/docs/configuring/services/gpu)

![Cover Photo: Enterprise Open-Weights AI Agents Built with Google Gemma, ADK & Cloud Run GPUs](./assets/article_cover_cloudguard_gemma.jpg)

**CloudGuard** is an end-to-end reference implementation for building, connecting, scaling, governing, evaluating, and QLoRA fine-tuning a production-ready **Enterprise AI Agent** on **Google Cloud** using **open-weights Gemma models** (`google/gemma-3-4b-it`, `google/gemma-3-12b-it`, `google/gemma-3-27b-it`) and the **Google Agent Development Kit (ADK)**.

![Figure 1: Gemma Enterprise Agent Platform Architecture](./assets/series_architecture_pillars.jpg)

---

## Architecture Overview

This repository implements the complete **6-Part Enterprise Agent Lifecycle**:

1. **Part 1 — Build ([`cloudguard_agent/agent.py`](./cloudguard_agent/agent.py)):** Model-agnostic `LlmAgent` using `LiteLlm` to seamlessly switch between local Ollama (`local_ollama`) for zero-cost laptop prototyping and self-hosted **vLLM on Cloud Run GPUs** (`cloud_run_vllm`) or **Vertex AI Model Garden** in production.
2. **Part 2 — Connect ([`cloudguard_agent/tools.py`](./cloudguard_agent/tools.py)):** Native Python tools with Gemma-optimized docstrings and type annotations backed by **Cloud Firestore** (plus a zero-config local JSON schema mirror for offline testing and CI/CD).
3. **Part 3 — Scale ([`deploy/`](./deploy/) & [`cloudguard_agent/memory.py`](./cloudguard_agent/memory.py)):** Serverless GPU deployment on **Cloud Run (`nvidia-l4` / `nvidia-rtx-pro-6000`)** using **Cloud Storage FUSE** for zero-download model streaming, paired with **Short-Term Session State** and a **Long-Term Enterprise Memory Bank**.
4. **Part 4 — Govern ([`cloudguard_agent/guardrails.py`](./cloudguard_agent/guardrails.py)):** Defense-in-depth security with private Cloud Run IAM service accounts, **Model Armor** input sanitization (redacting live GCP API keys `AIza...` and blocking prompt injections), and deterministic **RBAC Tool Policies** (`POL-PROD-MUTATION-001`).
5. **Part 5 — Evaluate & Observe ([`scripts/run_eval.py`](./scripts/run_eval.py)):** Automated **ADK Eval** test runner scoring **Tool Trajectory Accuracy (`1.00`)** and **Response Factuality (`1.00`)** while emitting **Google Cloud Trace** `gen_ai.*` OpenTelemetry spans.
6. **Part 6 — Fine-Tune ([`cloudguard_agent/finetuning.py`](./cloudguard_agent/finetuning.py)):** Converts verified multi-turn tool trajectories into Gemma `<start_of_turn>` SFT JSONL format and generates a **Vertex AI QLoRA (`NVIDIA_L4`)** training spec so a compact **Gemma 4B** model matches **Gemma 27B** tool-calling accuracy (`1.00`) at **2.8x faster latency and 6x lower VRAM**.

![Figure 2: End-to-End Request Flow](./assets/end_to_end_request_flow.jpg)

---

## Repository Structure

```text
cloudguard-gemma-agent/
├── pyproject.toml
├── .env.example
├── cloudguard_agent/
│   ├── __init__.py
│   ├── agent.py                  # Core ADK LlmAgent & runtime orchestrator
│   ├── tools.py                  # Native Firestore / FinOps Python tools
│   ├── memory.py                 # Session State & Enterprise Memory Bank
│   ├── guardrails.py             # Model Armor & RBAC policy callbacks
│   └── finetuning.py             # Vertex AI QLoRA SFT dataset & job builder
├── deploy/
│   ├── Dockerfile.vllm           # vLLM + GCS FUSE container for Cloud Run GPUs
│   ├── Dockerfile.vllm.lora      # Dynamic multi-LoRA serving container
│   └── deploy_cloud_run_gpu.sh   # Zero-trust IAM + Cloud Run GPU deploy script
├── scripts/
│   ├── seed_firestore.py         # Seeds incidents & GPU quota collections
│   ├── simulate_parts.py         # Runs Parts 1–4 multi-turn & security tests
│   ├── run_eval.py               # Runs Part 5 ADK Eval suite + OTel spans
│   └── prepare_and_verify_lora.py# Exports Part 6 Gemma SFT JSONL & benchmarks
└── tests/
    └── eval_dataset.json         # Golden evaluation dataset (5 core scenarios)
```

---

## Step-by-Step Architecture & Live Google Cloud Shell Verification

### Part 1: Launching in Google Cloud Shell

Click [**Open in Google Cloud Shell**](https://shell.cloud.google.com/?cloudshell_git_repo=https://github.com/RobinaMirbahar/cloudguard-gemma-agent) to clone and run the repository directly in your browser:

![Opening in Google Cloud Shell](./assets/open_in_cloud_shell.png)

![Authorizing Google Cloud Shell](./assets/cloud_shell_authorize.png)

![Cloning and running in Google Cloud Shell Terminal](./assets/cloud_shell_terminal_clone.png)

---

### Part 2: Native Python Tools & Cloud Firestore Seeding

![Figure 3: Part 2 — ADK Native Function Calling & Cloud Firestore Integration](./assets/part2_native_tools_firestore.jpg)

```bash
python3 scripts/seed_firestore.py
```

![Seeded incidents and regional GPU FinOps quota collections in Cloud Shell](./assets/cloud_shell_seed_db.png)

---

### Part 3: Serverless GPU Scaling (Cloud Run + vLLM + GCS FUSE) & Dual Memory

![Figure 4: Part 3 — Serverless GPU Scaling on Cloud Run & Dual Memory Architecture](./assets/part3_cloud_run_gpu_memory.jpg)

```bash
python3 scripts/simulate_parts.py
```

![Multi-turn incident triage, FinOps GPU quota check, and session-aware mitigation in Cloud Shell](./assets/cloud_shell_part2_part3_turns.png)

---

### Part 4: Defense-in-Depth Security (IAM, Model Armor & RBAC Callbacks)

![Figure 5: Part 4 — Defense-in-Depth Security with IAM, Model Armor & RBAC Callbacks](./assets/part4_zero_trust_governance.jpg)

![Model Armor API key redaction, RBAC production denial, and prompt injection blocking in Cloud Shell](./assets/cloud_shell_part4_guardrails.png)

---

### Part 5: Deterministic Evaluation (ADK Eval) & OpenTelemetry Tracing

![Figure 6: Part 5 — Deterministic CI/CD Evaluation (ADK Eval) & OpenTelemetry Tracing](./assets/part5_adk_eval_otel.jpg)

```bash
python3 scripts/run_eval.py
```

![ADK Deterministic Evaluation Suite (5/5 Passed) and OpenTelemetry Cloud Trace span in Cloud Shell](./assets/cloud_shell_adk_eval.png)

---

### Part 6: Fine-Tuning Gemma with QLoRA on Vertex AI for Domain Tool Calling

![Figure 7: Part 6 — Fine-Tuning Gemma with QLoRA on Vertex AI for Domain Tool Calling](./assets/part6_vertex_lora_finetuning.jpg)

```bash
python3 scripts/prepare_and_verify_lora.py
```

![Gemma SFT tool-calling training example exported in Cloud Shell](./assets/cloud_shell_sft_example.png)

![Vertex AI QLoRA CustomJob specification and post-tuning ADK Eval benchmark in Cloud Shell](./assets/cloud_shell_lora_benchmark.png)

---

## Deploying to Google Cloud Run Serverless GPUs

When ready to deploy to Google Cloud with an attached **NVIDIA L4 GPU** and **Cloud Storage FUSE** weight streaming:

```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"

bash deploy/deploy_cloud_run_gpu.sh
```

---

## License

Apache 2.0
