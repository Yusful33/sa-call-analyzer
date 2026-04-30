#!/usr/bin/env bash
# Create the EKS cluster and deploy id-pain, LiteLLM, and Gong.
# Requires: secret files (infra/k8s/secret.yaml, infra/k8s/litellm-secret.yaml, infra/k8s/gong-secret.yaml) and ECR images already pushed.

set -e
CLUSTER_NAME="${EKS_CLUSTER_NAME:-id-pain-demo}"
AWS_REGION="${AWS_REGION:-us-east-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
K8S_DIR="$REPO_ROOT/infra/k8s"

echo "Cluster: $CLUSTER_NAME, region: $AWS_REGION"

# Check secret files exist (they are not in git)
for f in "$K8S_DIR/secret.yaml" "$K8S_DIR/litellm-secret.yaml" "$K8S_DIR/gong-secret.yaml"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required file: $f (copy from the .example and fill in values)"
    exit 1
  fi
done

echo "Creating EKS cluster (this takes 15–25 min)..."
eksctl create cluster \
  --name "$CLUSTER_NAME" \
  --region "$AWS_REGION" \
  --nodegroup-name standard-workers \
  --node-type t3.micro \
  --nodes 10 \
  --nodes-min 0 \
  --nodes-max 10

echo "Updating kubeconfig..."
aws eks update-kubeconfig --region "$AWS_REGION" --name "$CLUSTER_NAME"

echo "Deploying apps..."
kubectl apply -f "$K8S_DIR/namespace.yaml"

kubectl apply -f "$K8S_DIR/litellm-configmap.yaml"
kubectl apply -f "$K8S_DIR/litellm-secret.yaml"
kubectl apply -f "$K8S_DIR/litellm-deployment.yaml"
kubectl apply -f "$K8S_DIR/litellm-service.yaml"

kubectl apply -f "$K8S_DIR/gong-secret.yaml"
kubectl apply -f "$K8S_DIR/gong-deployment.yaml"
kubectl apply -f "$K8S_DIR/gong-service.yaml"

kubectl apply -f "$K8S_DIR/configmap.yaml"
kubectl apply -f "$K8S_DIR/secret.yaml"
kubectl apply -f "$K8S_DIR/deployment.yaml"
kubectl apply -f "$K8S_DIR/service.yaml"

echo "Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=id-pain -n id-pain --timeout=300s || true
kubectl wait --for=condition=ready pod -l app=litellm -n id-pain --timeout=300s || true
kubectl wait --for=condition=ready pod -l app=gong-http-server -n id-pain --timeout=300s || true

echo "Done. Check: kubectl get pods -n id-pain"
echo "Access app: kubectl port-forward -n id-pain svc/id-pain 9090:80   then open http://localhost:9090"
