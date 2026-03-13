#!/usr/bin/env bash
# Scale the node group to 0 so no pods run. You stop paying for EC2; the cluster and all k8s resources stay.
# Cost when "down": ~$73/month (control plane only). To bring pods back: ./scripts/up-pods.sh

set -e
CLUSTER_NAME="${EKS_CLUSTER_NAME:-id-pain-demo}"
NODEGROUP_NAME="${EKS_NODEGROUP_NAME:-standard-workers}"
AWS_REGION="${AWS_REGION:-us-east-1}"

echo "Scaling node group to 0: cluster=$CLUSTER_NAME nodegroup=$NODEGROUP_NAME region=$AWS_REGION"
echo "Pods will go away (no nodes to run on). Cluster and Deployments/Secrets stay."
eksctl scale nodegroup --cluster="$CLUSTER_NAME" --name="$NODEGROUP_NAME" --nodes=0 --region="$AWS_REGION"
echo "Done. No worker nodes; no EC2 cost. To bring pods back: ./scripts/up-pods.sh"
