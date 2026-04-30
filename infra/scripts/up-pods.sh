#!/usr/bin/env bash
# Scale the node group back up so pods can run. Cluster and k8s resources already exist; no need to re-apply.
# Run this after down-pods.sh when you want the app available again.

set -e
CLUSTER_NAME="${EKS_CLUSTER_NAME:-id-pain-demo}"
NODEGROUP_NAME="${EKS_NODEGROUP_NAME:-standard-workers}"
AWS_REGION="${AWS_REGION:-us-east-1}"
NODES="${EKS_NODES:-10}"

echo "Scaling node group to $NODES: cluster=$CLUSTER_NAME nodegroup=$NODEGROUP_NAME region=$AWS_REGION"
eksctl scale nodegroup --cluster="$CLUSTER_NAME" --name="$NODEGROUP_NAME" --nodes="$NODES" --region="$AWS_REGION"

echo "Waiting for nodes to be ready..."
kubectl wait --for=condition=ready nodes -l "eks.amazonaws.com/nodegroup=$NODEGROUP_NAME" --timeout=300s 2>/dev/null || true

echo "Pods should be scheduling. Check: kubectl get pods -n id-pain"
echo "Access app: kubectl port-forward -n id-pain svc/id-pain 9090:80   then open http://localhost:9090"
