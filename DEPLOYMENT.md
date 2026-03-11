# Deploying id-pain (SA Call Analyzer) on AWS with Kubernetes

This guide covers deploying **id-pain** (the SA Call Analyzer CrewAI app) from local/Railway to **AWS EKS** and setting up a **Grafana** demo.

---

## 1. Prerequisites

- **AWS CLI** configured (`aws configure`)
- **kubectl** installed
- **Docker** (for building/pushing images)
- **eksctl** (recommended) or Terraform / AWS Console for EKS

---

## 2. Build and push images to Amazon ECR

You have three images: **id-pain** (main app), **litellm** (LLM proxy), and **gong-http-server** (Gong MCP). Create an ECR repo for each and push from the correct build context.

```bash
# Set your AWS account and region
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1
export ECR_URI=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# One-time: create ECR repositories
aws ecr create-repository --repository-name id-pain --region $AWS_REGION
aws ecr create-repository --repository-name litellm --region $AWS_REGION
aws ecr create-repository --repository-name gong-http-server --region $AWS_REGION

# Login once
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URI
```

**Build and push each image** (run from the repo root `id-pain/`):

```bash
# 1. id-pain (main app)
docker build -t id-pain:latest .
docker tag id-pain:latest $ECR_URI/id-pain:latest
docker push $ECR_URI/id-pain:latest

# 2. LiteLLM (build from litellm/ subdirectory)
docker build -t litellm:latest ./litellm
docker tag litellm:latest $ECR_URI/litellm:latest
docker push $ECR_URI/litellm:latest

# 3. Gong HTTP server (build from gong-http-server/ subdirectory)
docker build -t gong-http-server:latest ./gong-http-server
docker tag gong-http-server:latest $ECR_URI/gong-http-server:latest
docker push $ECR_URI/gong-http-server:latest
```

---

## 3. Create an EKS cluster (if you don’t have one)

**Option A – eksctl**

```bash
eksctl create cluster \
  --name id-pain-demo \
  --region $AWS_REGION \
  --nodegroup-name standard-workers \
  --node-type t3.small \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 3
```

**Option B – AWS Console**  
Create cluster via EKS → Create cluster (Fargate or managed node groups).

Then:

```bash
aws eks update-kubeconfig --region $AWS_REGION --name id-pain-demo
```

---

## 4. Kubernetes manifests

Manifests live in **`k8s/`**. All three apps run in the **`id-pain`** namespace. id-pain is wired to LiteLLM at `http://litellm:4000` and to the Gong MCP server at `http://gong-http-server:8080` via the id-pain ConfigMap.

| Component          | ConfigMap / Secret      | Key env vars |
|-------------------|-------------------------|--------------|
| **id-pain**       | id-pain-config, id-pain-secrets | PORT, MODEL_NAME, USE_LITELLM, LITELLM_BASE_URL, GONG_MCP_URL, ARIZE_*, ANTHROPIC_*, OPENAI_*, GCP_CREDENTIALS_BASE64 |
| **LiteLLM**       | litellm-config, litellm-secrets | PORT, ANTHROPIC_API_KEY, OPENAI_API_KEY |
| **Gong HTTP**     | gong-secrets            | GONG_ACCESS_KEY, GONG_SECRET_KEY |

**GCP credentials:** The id-pain Dockerfile’s `start.sh` supports **`GCP_CREDENTIALS_BASE64`** (base64-encoded service account JSON). Store that in `id-pain-secrets` so the container writes credentials at startup.

### 4.1 One-time secret setup

Copy each example to a real secret file, fill in values, then apply (do not commit the real secret files):

```bash
# id-pain (main app + Arize, GCP, etc.)
cp k8s/secret.yaml.example k8s/secret.yaml
# Edit k8s/secret.yaml: ARIZE_*, ANTHROPIC_*, OPENAI_*, GONG_* (optional for id-pain), GCP_CREDENTIALS_BASE64
kubectl apply -f k8s/secret.yaml

# LiteLLM (needs LLM provider keys; config.yaml reads os.environ/ANTHROPIC_API_KEY, OPENAI_API_KEY)
cp k8s/litellm-secret.yaml.example k8s/litellm-secret.yaml
# Edit k8s/litellm-secret.yaml
kubectl apply -f k8s/litellm-secret.yaml

# Gong HTTP server (Gong API credentials)
cp k8s/gong-secret.yaml.example k8s/gong-secret.yaml
# Edit k8s/gong-secret.yaml
kubectl apply -f k8s/gong-secret.yaml
```

To generate **GCP_CREDENTIALS_BASE64**:

```bash
base64 -w0 /path/to/your-service-account.json
# Paste into k8s/secret.yaml under GCP_CREDENTIALS_BASE64
```

### 4.2 Deploy (order matters)

Replace `<AWS_ACCOUNT_ID>` and `<AWS_REGION>` in **all** deployment YAMLs (id-pain, litellm, gong). You can use:

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1
```

Then apply in this order (LiteLLM and Gong first so id-pain can reach them):

```bash
# Namespace once
kubectl apply -f k8s/namespace.yaml

# LiteLLM
kubectl apply -f k8s/litellm-configmap.yaml
kubectl apply -f k8s/litellm-secret.yaml
sed -e "s/<AWS_ACCOUNT_ID>/$AWS_ACCOUNT_ID/g" -e "s/<AWS_REGION>/$AWS_REGION/g" k8s/litellm-deployment.yaml | kubectl apply -f -
kubectl apply -f k8s/litellm-service.yaml

# Gong HTTP server
kubectl apply -f k8s/gong-secret.yaml
sed -e "s/<AWS_ACCOUNT_ID>/$AWS_ACCOUNT_ID/g" -e "s/<AWS_REGION>/$AWS_REGION/g" k8s/gong-deployment.yaml | kubectl apply -f -
kubectl apply -f k8s/gong-service.yaml

# id-pain (main app)
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
sed -e "s/<AWS_ACCOUNT_ID>/$AWS_ACCOUNT_ID/g" -e "s/<AWS_REGION>/$AWS_REGION/g" k8s/deployment.yaml | kubectl apply -f -
kubectl apply -f k8s/service.yaml

# Optional: expose id-pain via ALB
kubectl apply -f k8s/ingress.yaml
```

### 4.3 Verify

```bash
kubectl get pods,svc -n id-pain
# Expect: id-pain, litellm, gong-http-server pods and services

kubectl logs -f deployment/id-pain -n id-pain
kubectl logs -f deployment/litellm -n id-pain
kubectl logs -f deployment/gong-http-server -n id-pain
```

Port-forward for local testing:

```bash
kubectl port-forward -n id-pain svc/id-pain 8080:80
# Open http://localhost:8080 and http://localhost:8080/health
```

---

## 5. Grafana demo setup

### 5.1 What to show

- **App health and traffic:** request rate, latency, errors for the id-pain API.
- **Link to Arize:** “Traces and LLM visibility live in Arize” from a dashboard panel.

### 5.2 Option A – Prometheus + Grafana in-cluster

1. Install [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack) (includes Prometheus + Grafana).
2. Expose **/metrics** from the FastAPI app (e.g. `prometheus-fastapi-instrumentator`) and configure Prometheus to scrape the `id-pain` service.
3. In Grafana, create a dashboard with panels for request rate, latency, error rate, and (optional) pod CPU/memory.

### 5.3 Option B – Grafana Cloud

1. Sign up at [grafana.com/products/cloud](https://grafana.com/products/cloud/).
2. Use Grafana Cloud Prometheus and deploy the Grafana Agent (or a scraper) in the cluster to push metrics.
3. Build the same style of dashboard in Grafana Cloud.

### 5.4 Demo tips

- **Pre-build one dashboard** (e.g. “id-pain – SA Call Analyzer”) with 3–4 panels so the demo starts with a clear narrative.
- **Add a link to Arize** (link panel or text) to your Arize project so you can say: “Metrics here, full traces and LLM visibility in Arize.”
- **Generate light load** before the demo (e.g. hit `/health` or a simple `/api/` endpoint) so graphs aren’t empty.
- **Optional:** Add one alert (e.g. error rate or latency) and show it in the UI.

### 5.5 Optional: Prometheus metrics in the app

To get app-level metrics (request count, latency) into Grafana:

```bash
pip install prometheus-client prometheus-fastapi-instrumentator
```

In `main.py`, after `app = FastAPI(...)`:

```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
```

Then point Prometheus at the Service and path `/metrics`, and use the metrics in Grafana.

---

## 6. Summary checklist

| Step | Action |
|------|--------|
| 1 | Build and push **id-pain**, **litellm**, and **gong-http-server** images to ECR (Section 2) |
| 2 | Create or use an EKS cluster; update kubeconfig |
| 3 | Create secrets from examples: `secret.yaml`, `litellm-secret.yaml`, `gong-secret.yaml`; fill in and apply |
| 4 | Apply `k8s/` manifests in order: namespace → LiteLLM → Gong → id-pain (Section 4.2) |
| 5 | (Optional) Add Ingress for a public URL to id-pain |
| 6 | (Optional) Install Prometheus + Grafana; add `/metrics` and a dashboard; link to Arize |

---

## 7. Railway vs AWS/Kubernetes

| Aspect | Railway | AWS EKS |
|--------|---------|---------|
| Build | Same Dockerfile | Same image, pushed to ECR |
| Env | Railway dashboard / CLI | ConfigMap + Secret |
| GCP | `GCP_CREDENTIALS_BASE64` | Same – put in K8s Secret |
| Health | `/health` (Railway healthcheck) | Same – use in liveness/readiness probes |
| Port | `PORT` (e.g. 8080) | Same – container listens on `PORT`, Service exposes 80 |

No code changes are required; the same Dockerfile and startup script work on both Railway and EKS.
