#!/usr/bin/env bash
# ============================================================================
# Part 3 & Part 4 Deployment Script:
# Deploys Open-Weights Gemma on Cloud Run Serverless GPUs + Cloud Storage FUSE
# with Zero-Trust IAM Service Account Isolation
# ============================================================================
set -euo pipefail

export PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-cloudguard-enterprise-demo}"
export REGION="${GOOGLE_CLOUD_LOCATION:-us-central1}"
export MODEL_BUCKET="${PROJECT_ID}-gemma-weights"
export VLLM_SA="gemma-vllm-sa@${PROJECT_ID}.iam.gserviceaccount.com"
export AGENT_SA="cloudguard-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com"

echo "[1/3] Deploying Gemma 27B vLLM Inference Service on Cloud Run with NVIDIA L4 GPU..."
gcloud run deploy gemma-vllm-service \
  --image="us-docker.pkg.dev/${PROJECT_ID}/cloudguard-repo/gemma-vllm:latest" \
  --region="${REGION}" \
  --service-account="${VLLM_SA}" \
  --no-allow-unauthenticated \
  --cpu=8 \
  --memory=32Gi \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --min-instances=0 \
  --max-instances=3 \
  --concurrency=8 \
  --add-volume="name=model-weights,type=cloud-storage,bucket=${MODEL_BUCKET},readonly=true" \
  --add-volume-mount="volume=model-weights,mount-path=/mnt/models" \
  --port=8080

VLLM_URL=$(gcloud run services describe gemma-vllm-service --region="${REGION}" --format="value(status.url)")
echo "[2/3] Granting CloudGuard Agent SA permission to invoke Gemma GPU endpoint..."
gcloud run services add-iam-policy-binding gemma-vllm-service \
  --region="${REGION}" \
  --member="serviceAccount:${AGENT_SA}" \
  --role="roles/run.invoker"

echo "[3/3] Deploying ADK CloudGuard Agent Orchestrator to Cloud Run..."
gcloud run deploy cloudguard-adk-agent \
  --source=. \
  --region="${REGION}" \
  --service-account="${AGENT_SA}" \
  --no-allow-unauthenticated \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GEMMA_BACKEND_MODE=cloud_run_vllm,GEMMA_VLLM_ENDPOINT_URL=${VLLM_URL}/v1,USE_LOCAL_MOCK_DB=false"

echo "Deployment Complete! Private Gemma Endpoint: ${VLLM_URL}"
