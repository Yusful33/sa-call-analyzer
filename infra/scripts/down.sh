#!/usr/bin/env bash
# Tear down the EKS cluster to stop paying for it.
# Your state is preserved: ECR images, k8s YAML, and secret files on disk (see DEPLOYMENT.md §8).

set -e
CLUSTER_NAME="${EKS_CLUSTER_NAME:-id-pain-demo}"
AWS_REGION="${AWS_REGION:-us-east-1}"

echo "Deleting EKS cluster: $CLUSTER_NAME (region: $AWS_REGION)"
echo "This takes a few minutes. State (images, manifests, secrets) is not lost."
read -p "Continue? [y/N] " -n 1 -r; echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 1
fi

eksctl delete cluster --name "$CLUSTER_NAME" --region "$AWS_REGION"
echo "Cluster deleted. To bring it back, run: infra/scripts/up.sh (from repo root)."
