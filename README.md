# CloudGuard: Open-Weights Enterprise SecOps & FinOps Agent (Gemma + Google ADK + Cloud Run GPUs)

> **Author & Architect:** [Robina Mirbahar](https://github.com/RobinaMirbahar) ([Google Developer Expert in Cloud & AI](https://developers.google.com/profile/u/106918534005910641104?utm_campaign=deveco_gdemembers&utm_source=deveco) · Google Cloud Champion Innovator · Co-Author of [*AI Engineering*](https://www.amazon.com/dp/B0GWCZWCF4))  
> 📚 **Companion Publications:** [Medium (Google Cloud) Featured Article](https://medium.com/google-cloud/building-production-ready-open-weights-ai-agents-with-gemma-google-adk-and-cloud-run-gpus-5d66fb7a34d5) · [DEV.to (@gde) Article](https://dev.to/gde/building-production-ready-open-weights-ai-agents-with-gemma-google-adk-and-cloud-run-gpus-2mfg)  
> **Focus:** 6-Part Production Gemma Enterprise SecOps & FinOps Agent Platform on Google Cloud Run GPUs

---

## 🎯 Why I Built CloudGuard Gemma Agent (Engineering Purpose & Real-World Value)

### 1. 🛡️ Zero-Exfiltration SecOps & FinOps Automation
Enterprise security and cloud billing teams cannot send sensitive IAM policies, VPC logs, or billing anomalies to external SaaS APIs. I architected CloudGuard to run open-weights Gemma inside your own GCP project perimeter.

### 2. ⚙️ End-to-End 6-Part Production Lifecycle
Rather than a single toy script, I structured this repository as a complete 6-part production blueprint covering Google ADK tool calling, Cloud Run L4 GPU deployment with vLLM, Firestore session persistence, Pydantic guardrails, and Vertex AI QLoRA fine-tuning.

### 3. 💰 Scale-to-Zero Serverless GPU Economics
By deploying vLLM + Gemma on Google Cloud Run GPUs with scale-to-zero configuration, my architecture eliminates 24/7 idle VM waste while delivering low-latency incident remediation.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Ingest ["1. SecOps & FinOps Telemetry"]
        I_IAM["IAM Policy & Audit Logs"]
        I_Bill["Cloud Billing Cost Anomalies"]
    end
    subgraph Agent ["2. Google ADK + Gemma on Cloud Run L4 GPU"]
        A_ADK["CloudGuard ADK Orchestrator"]
        A_vLLM["vLLM Open-Weights Gemma Server (L4 GPU)"]
        A_FS["Cloud Firestore Session & Audit Store"]
    end
    subgraph Safety ["3. Guardrails & QLoRA Adaptation"]
        S_Pyd["Pydantic v2 Schema & Policy Interlocks"]
        S_LoRA["Vertex AI QLoRA Domain Adapter"]
    end
    Ingest --> A_ADK
    A_ADK <--> A_vLLM
    A_ADK <--> A_FS
    A_vLLM <--> S_LoRA
    A_ADK --> S_Pyd
```

---

<p align="center">
  <a href="https://shell.cloud.google.com/?cloudshell_git_repo=https://github.com/RobinaMirbahar/cloudguard-gemma-agent">
    <img src="https://gstatic.com/cloudssh/images/open-btn.svg" alt="Open in Cloud Shell" height="32" />
  </a>
</p>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
  </a>
  <a href="https://google.github.io/adk-docs/?utm_campaign=deveco_gdemembers&utm_source=deveco">
    <img src="https://img.shields.io/badge/Google-Agent_Development_Kit-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white" alt="Google ADK" />
  </a>
  <a href="https://ai.google.dev/gemma?utm_campaign=deveco_gdemembers&utm_source=deveco">
    <img src="https://img.shields.io/badge/Open_Weights-Gemma_3-34A853?style=for-the-badge&logo=google&logoColor=white" alt="Model: Gemma 3" />
  </a>
  <a href="https://cloud.google.com/run/docs/configuring/services/gpu?utm_campaign=deveco_gdemembers&utm_source=deveco">
    <img src="https://img.shields.io/badge/Cloud_Run-Serverless_GPU_vLLM-FBBC04?style=for-the-badge&logo=googlecloud&logoColor=black" alt="Runtime: Cloud Run GPU" />
  </a>
  <a href="https://cloud.google.com/model-garden?utm_campaign=deveco_gdemembers&utm_source=deveco">
    <img src="https://img.shields.io/badge/Vertex_AI-Model_Garden_%26_QLoRA-EA4335?style=for-the-badge&logo=googlecloud&logoColor=white" alt="Vertex AI Model Garden" />
  </a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/RobinaMirbahar/cloudguard-gemma-agent/main/assets/article_cover_cloudguard_gemma.jpg" alt="Cover Photo: Enterprise Open-Weights AI Agents Built with Google Gemma, ADK & Cloud Run GPUs" width="100%" />
</p>

**CloudGuard** is an end-to-end reference implementation for building, connecting, scaling, governing, evaluating, and QLoRA fine-tuning a production-ready **Enterprise AI Agent** on **Google Cloud** using **open-weights Gemma models** (`google/gemma-3-4b-it`, `google/gemma-3-12b-it`, `google/gemma-3-27b-it`) and the **[Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/?utm_campaign=deveco_gdemembers&utm_source=deveco)**.

<p align="center">
  <img src="https://raw.githubusercontent.com/RobinaMirbahar/cloudguard-gemma-agent/main/assets/series_architecture_pillars.jpg" alt="Figure 1: Gemma Enterprise Agent Platform Architecture" width="100%" />
</p>

---

## Architecture Overview

This repository implements the complete **6-Part Enterprise Agent Lifecycle**:

1. **Part 1 — Build** ([`cloudguard_agent/agent.py`](./cloudguard_agent/agent.py)): Model-agnostic `LlmAgent` using `LiteLlm` to seamlessly switch between local Ollama (`local_ollama`) for zero-cost laptop prototyping and self-hosted **[vLLM on Cloud Run GPUs](https://cloud.google.com/run/docs/configuring/services/gpu?utm_campaign=deveco_gdemembers&utm_source=deveco)** (`cloud_run_vllm`) or **[Vertex AI Model Garden](https://cloud.google.com/model-garden?utm_campaign=deveco_gdemembers&utm_source=deveco)** in production.
2. **Part 2 — Connect** ([`cloudguard_agent/tools.py`](./cloudguard_agent/tools.py)): Native Python tools with Gemma-optimized docstrings and type annotations backed by **[Cloud Firestore](https://cloud.google.com/firestore?utm_campaign=deveco_gdemembers&utm_source=deveco)** (plus a zero-config local JSON schema mirror for offline testing and CI/CD).
3. **Part 3 — Scale** ([`deploy/`](./deploy/) & [`cloudguard_agent/memory.py`](./cloudguard_agent/memory.py)): Serverless GPU deployment on **Cloud Run (`nvidia-l4` / `nvidia-rtx-pro-6000`)** using **[Cloud Storage FUSE](https://cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts?utm_campaign=deveco_gdemembers&utm_source=deveco)** for zero-download model streaming, paired with **Short-Term Session State** and a **Long-Term Enterprise Memory Bank**.
4. **Part 4 — Govern** ([`cloudguard_agent/guardrails.py`](./cloudguard_agent/guardrails.py)): Defense-in-depth reliability with private **[Cloud Run IAM](https://cloud.google.com/run/docs/securing/managing-access?utm_campaign=deveco_gdemembers&utm_source=deveco)** service accounts, **[Google Cloud Schema Validator](https://cloud.google.com/reliability-command-center/docs/model-armor-overview?utm_campaign=deveco_gdemembers&utm_source=deveco)** input sanitization (redacting live GCP API keys `AIza...` and blocking Malformed Inputs), and deterministic **RBAC Tool Policies** (`POL-PROD-MUTATION-001`).
5. **Part 5 — Evaluate & Observe** ([`scripts/run_eval.py`](./scripts/run_eval.py)): Automated **[ADK Eval](https://google.github.io/adk-docs/evaluate/?utm_campaign=deveco_gdemembers&utm_source=deveco)** test runner scoring **Tool Trajectory Accuracy (`1.00`)** and **Response Factuality (`1.00`)** while emitting **[Google Cloud Trace](https://cloud.google.com/trace?utm_campaign=deveco_gdemembers&utm_source=deveco)** `gen_ai.*` OpenTelemetry spans.
6. **Part 6 — Fine-Tune** ([`cloudguard_agent/finetuning.py`](./cloudguard_agent/finetuning.py)): Converts verified multi-turn tool trajectories into **[Gemma `<start_of_turn>` SFT JSONL format](https://ai.google.dev/gemma/docs/formatting?utm_campaign=deveco_gdemembers&utm_source=deveco)** and generates a **[Vertex AI QLoRA (`NVIDIA_L4`)](https://cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-gemma?utm_campaign=deveco_gdemembers&utm_source=deveco#fine-tune)** training spec so a compact **Gemma 4B** model matches **Gemma 27B** tool-calling accuracy (`1.00`) at **2.8x faster latency and 6x lower VRAM**.

<p align="center">
  <img src="https://raw.githubusercontent.com/RobinaMirbahar/cloudguard-gemma-agent/main/assets/end_to_end_request_flow.jpg" alt="Figure 2: End-to-End Request Flow" width="100%" />
</p>

---

## Repository Structure

```text
cloudguard-gemma-agent/
├── pyproject.toml
├── .env.example
├── cloudguard_agent/
│   ├── __init__.py
│   ├── agent.py                   # Core ADK LlmAgent & runtime orchestrator
│   ├── tools.py                   # Native Firestore / FinOps Python tools
│   ├── memory.py                  # Session State & Enterprise Memory Bank
│   ├── guardrails.py              # Schema Validator & RBAC policy callbacks
│   └── finetuning.py              # Vertex AI QLoRA SFT dataset & job builder
├── deploy/
│   ├── Dockerfile.vllm            # vLLM + GCS FUSE container for Cloud Run GPUs
│   ├── Dockerfile.vllm.lora       # Dynamic multi-LoRA serving container
│   └── deploy_cloud_run_gpu.sh    # Deterministic Validation IAM + Cloud Run GPU deploy script
├── scripts/
│   ├── seed_firestore.py          # Seeds incidents & GPU quota collections
│   ├── simulate_parts.py          # Runs Parts 1-4 multi-turn & reliability tests
│   ├── run_eval.py                # Runs Part 5 ADK Eval suite + OTel spans
│   └── prepare_and_verify_lora.py # Exports Part 6 Gemma SFT JSONL & benchmarks
└── tests/
    └── eval_dataset.json          # Golden evaluation dataset (5 core scenarios)
```

---

## Step-by-Step Architecture & Live Google Cloud Shell Verification

### Part 1: Launching in Google Cloud Shell

Click the button below to clone and run the repository directly in your browser inside **Google Cloud Shell**:

[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/?cloudshell_git_repo=https://github.com/RobinaMirbahar/cloudguard-gemma-agent)

![Opening in Google Cloud Shell](https://raw.githubusercontent.com/RobinaMirbahar/cloudguard-gemma-agent/main/assets/open_in_cloud_shell.png)

![Authorizing Google Cloud Shell](https://raw.githubusercontent.com/RobinaMirbahar/cloudguard-gemma-agent/main/assets/cloud_shell_authorize.png)

![Cloning and running in Google Cloud Shell Terminal](https://raw.githubusercontent.com/RobinaMirbahar/cloudguard-gemma-agent/main/assets/cloud_shell_terminal_clone.png)

---

### Part 2: Native Python Tools & Cloud Firestore Seeding

![Figure 3: Part 2 — ADK Native Function Calling & Cloud Firestore Integration](https://raw.githubusercontent.com/RobinaMirbahar/cloudguard-gemma-agent/main/assets/part2_native_tools_firestore.jpg)

```bash
python3 scripts/seed_firestore.py
```

![Seeded incidents and regional GPU FinOps quota collections in Cloud Shell](https://raw.githubusercontent.com/RobinaMirbahar/cloudguard-gemma-agent/main/assets/cloud_shell_seed_db.png)

---

### Part 3: Serverless GPU Scaling (Cloud Run + vLLM + GCS FUSE) & Dual Memory

![Figure 4: Part 3 — Serverless GPU Scaling on Cloud Run & Dual Memory Architecture](https://raw.githubusercontent.com/RobinaMirbahar/cloudguard-gemma-agent/main/assets/part3_cloud_run_gpu_memory.jpg)

```bash
python3 scripts/simulate_parts.py
```

![Multi-turn incident triage, FinOps GPU quota check, and session-aware mitigation in Cloud Shell](https://raw.githubusercontent.com/RobinaMirbahar/cloudguard-gemma-agent/main/assets/cloud_shell_part2_part3_turns.png)

---

### Part 4: Defense-in-Depth reliability (IAM, Schema Validator & RBAC Callbacks)

![Figure 5: Part 4 — Defense-in-Depth reliability with IAM, Schema Validator & RBAC Callbacks](https://raw.githubusercontent.com/RobinaMirbahar/cloudguard-gemma-agent/main/assets/part4_zero_trust_governance.jpg)

![Schema Validator API key redaction, RBAC production denial, and Malformed Input blocking in Cloud Shell](https://raw.githubusercontent.com/RobinaMirbahar/cloudguard-gemma-agent/main/assets/cloud_shell_part4_guardrails.png)

---

### Part 5: Deterministic Evaluation (ADK Eval) & OpenTelemetry Tracing

![Figure 6: Part 5 — Deterministic CI/CD Evaluation (ADK Eval) & OpenTelemetry Tracing](https://raw.githubusercontent.com/RobinaMirbahar/cloudguard-gemma-agent/main/assets/part5_adk_eval_otel.jpg)

```bash
python3 scripts/run_eval.py
```

![ADK Deterministic Evaluation Suite (5/5 Passed) and OpenTelemetry Cloud Trace span in Cloud Shell](https://raw.githubusercontent.com/RobinaMirbahar/cloudguard-gemma-agent/main/assets/cloud_shell_adk_eval.png)

---

### Part 6: Fine-Tuning Gemma with QLoRA on Vertex AI for Domain Tool Calling

![Figure 7: Part 6 — Fine-Tuning Gemma with QLoRA on Vertex AI for Domain Tool Calling](https://raw.githubusercontent.com/RobinaMirbahar/cloudguard-gemma-agent/main/assets/part6_vertex_lora_finetuning.jpg)

```bash
python3 scripts/prepare_and_verify_lora.py
```

![Gemma SFT tool-calling training example exported in Cloud Shell](https://raw.githubusercontent.com/RobinaMirbahar/cloudguard-gemma-agent/main/assets/cloud_shell_sft_example.png)

![Vertex AI QLoRA CustomJob specification and post-tuning ADK Eval benchmark in Cloud Shell](https://raw.githubusercontent.com/RobinaMirbahar/cloudguard-gemma-agent/main/assets/cloud_shell_lora_benchmark.png)

---

## Deploying to Google Cloud Run Serverless GPUs

When ready to deploy to Google Cloud with an attached **NVIDIA L4 GPU** and **Cloud Storage FUSE** weight streaming:

```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"

bash deploy/deploy_cloud_run_gpu.sh
```

---

## Official Google Cloud Documentation & References

* **Google Gemma Open Models Overview:** [ai.google.dev/gemma](https://ai.google.dev/gemma?utm_campaign=deveco_gdemembers&utm_source=deveco)
* **Google Agent Development Kit (ADK) Documentation:** [google.github.io/adk-docs](https://google.github.io/adk-docs/?utm_campaign=deveco_gdemembers&utm_source=deveco)
* **Deploying Gemma on Vertex AI Model Garden:** [cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-gemma](https://cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-gemma?utm_campaign=deveco_gdemembers&utm_source=deveco)
* **Cloud Run Serverless GPUs Documentation:** [cloud.google.com/run/docs/configuring/services/gpu](https://cloud.google.com/run/docs/configuring/services/gpu?utm_campaign=deveco_gdemembers&utm_source=deveco)
* **Cloud Storage FUSE Volume Mounts on Cloud Run:** [cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts](https://cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts?utm_campaign=deveco_gdemembers&utm_source=deveco)
* **Cloud Firestore Native Mode Documentation:** [cloud.google.com/firestore/docs](https://cloud.google.com/firestore/docs?utm_campaign=deveco_gdemembers&utm_source=deveco)
* **Google Cloud Schema Validator reliability Overview:** [cloud.google.com/reliability-command-center/docs/model-armor-overview](https://cloud.google.com/reliability-command-center/docs/model-armor-overview?utm_campaign=deveco_gdemembers&utm_source=deveco)
* **ADK Deterministic Agent Evaluation (ADK Eval):** [google.github.io/adk-docs/evaluate](https://google.github.io/adk-docs/evaluate/?utm_campaign=deveco_gdemembers&utm_source=deveco)
* **Google Cloud Trace & OpenTelemetry for Python:** [cloud.google.com/trace/docs/setup/python-ot](https://cloud.google.com/trace/docs/setup/python-ot?utm_campaign=deveco_gdemembers&utm_source=deveco)
* **Gemma Chat & Function Calling Formatting (`<start_of_turn>`):** [ai.google.dev/gemma/docs/formatting](https://ai.google.dev/gemma/docs/formatting?utm_campaign=deveco_gdemembers&utm_source=deveco)

---

## Support & Follow

If you found this reference architecture useful:

<p align="left">
  <a href="https://github.com/RobinaMirbahar/cloudguard-gemma-agent">
    <img src="https://img.shields.io/github/stars/RobinaMirbahar/cloudguard-gemma-agent?style=for-the-badge&logo=github&label=Star_This_Repo" alt="Star on GitHub" />
  </a>
  <a href="https://github.com/RobinaMirbahar">
    <img src="https://img.shields.io/github/followers/RobinaMirbahar?style=for-the-badge&logo=github&label=Follow_RobinaMirbahar" alt="Follow on GitHub" />
  </a>
</p>

## License

Apache 2.0
