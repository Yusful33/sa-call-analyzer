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
  --node-type t3.micro \
  --nodes 10 \
  --nodes-min 0 \
  --nodes-max 10
```

When not in use, scale the node group to 0 (`./scripts/down-pods.sh`) to stop EC2 cost; scale back to 10 (`./scripts/up-pods.sh`) when you need the app.

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

Deploy in this order so that **LiteLLM** and **Gong** are running before **id-pain** starts. id-pain connects to them by Kubernetes service name (`http://litellm:4000`, `http://gong-http-server:8080`); if those services don’t exist yet, id-pain may log connection errors until they’re ready.

If your deployment YAMLs still use placeholders, set these so the `sed` commands work (or skip `sed` and apply the YAMLs directly if you’ve already filled in the image URLs):

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1
```

Then run the steps below. Each block explains what the step does.

---

**1. Create the namespace**

All resources (id-pain, LiteLLM, Gong) run in a single namespace so they can talk to each other by service name.

```bash
kubectl apply -f k8s/namespace.yaml
```

- **What it does:** Creates the `id-pain` namespace. Everything else (ConfigMaps, Secrets, Deployments, Services) is created inside this namespace.

---

**2. Deploy LiteLLM (LLM proxy)**

LiteLLM runs first so id-pain can send LLM requests to it once id-pain is deployed.

```bash
kubectl apply -f k8s/litellm-configmap.yaml
kubectl apply -f k8s/litellm-secret.yaml
sed -e "s/<AWS_ACCOUNT_ID>/$AWS_ACCOUNT_ID/g" -e "s/<AWS_REGION>/$AWS_REGION/g" k8s/litellm-deployment.yaml | kubectl apply -f -
kubectl apply -f k8s/litellm-service.yaml
```

- **litellm-configmap.yaml** – Non-sensitive config (e.g. `PORT=4000`) injected into the LiteLLM pods as environment variables.
- **litellm-secret.yaml** – API keys (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) used by LiteLLM’s `config.yaml` via `os.environ/...`. Stored as a Secret so they aren’t in plain YAML.
- **litellm-deployment.yaml** – Defines the LiteLLM workload: which ECR image to run, how many replicas, resource limits, and liveness/readiness probes. The `sed` replaces `<AWS_ACCOUNT_ID>` and `<AWS_REGION>` in the image URL (e.g. `574016082345.dkr.ecr.us-east-1.amazonaws.com/litellm:latest`) so Kubernetes pulls from your ECR. If the file already has real values, you can run `kubectl apply -f k8s/litellm-deployment.yaml` instead.
- **litellm-service.yaml** – Creates a ClusterIP Service named `litellm` on port 4000. Other pods in the namespace (e.g. id-pain) reach LiteLLM at `http://litellm:4000`.

---

**3. Deploy Gong HTTP server (Gong MCP)**

The Gong server runs next so id-pain can call it for Gong-related features.

```bash
kubectl apply -f k8s/gong-secret.yaml
sed -e "s/<AWS_ACCOUNT_ID>/$AWS_ACCOUNT_ID/g" -e "s/<AWS_REGION>/$AWS_REGION/g" k8s/gong-deployment.yaml | kubectl apply -f -
kubectl apply -f k8s/gong-service.yaml
```

- **gong-secret.yaml** – Gong API credentials (`GONG_ACCESS_KEY`, `GONG_SECRET_KEY`) used by the Gong MCP server to call the Gong API. No ConfigMap is needed; only these env vars are required.
- **gong-deployment.yaml** – Defines the Gong HTTP server workload (ECR image, replicas, probes). The `sed` again fills in the ECR image URL for your account and region.
- **gong-service.yaml** – Creates a ClusterIP Service named `gong-http-server` on port 8080. id-pain uses `GONG_MCP_URL=http://gong-http-server:8080` (from the id-pain ConfigMap) to talk to this service.

---

**4. Deploy id-pain (main app)**

With LiteLLM and Gong running, deploy the main app so it can use them.

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
sed -e "s/<AWS_ACCOUNT_ID>/$AWS_ACCOUNT_ID/g" -e "s/<AWS_REGION>/$AWS_REGION/g" k8s/deployment.yaml | kubectl apply -f -
kubectl apply -f k8s/service.yaml
```

- **configmap.yaml** – Non-sensitive id-pain config: `PORT`, `MODEL_NAME`, `USE_LITELLM`, `LITELLM_BASE_URL` (points to the LiteLLM service), `GONG_MCP_URL` (points to the Gong service), and CrewAI/OTEL toggles.
- **secret.yaml** – Sensitive id-pain config: `ARIZE_API_KEY`, `ARIZE_SPACE_ID`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, optional `GONG_*`, and `GCP_CREDENTIALS_BASE64` for BigQuery. The container’s startup script uses `GCP_CREDENTIALS_BASE64` to write a credentials file at runtime.
- **deployment.yaml** – Defines the id-pain workload (ECR image, env from ConfigMap + Secret, resources, health checks). `sed` fills in the ECR image URL; if already set, use `kubectl apply -f k8s/deployment.yaml` directly.
- **service.yaml** – Creates a ClusterIP Service for id-pain (port 80 → container 8080). This is what you port-forward to or what the Ingress targets.

---

**5. (Optional) Expose id-pain via an Application Load Balancer**

Only run this if you have the [AWS Load Balancer Controller](https://kubernetes-sigs.github.io/aws-load-balancer-controller/) installed in the cluster and want to give id-pain a public URL.

```bash
kubectl apply -f k8s/ingress.yaml
```

- **ingress.yaml** – Creates an Ingress that tells the AWS Load Balancer Controller to create an ALB and route traffic to the id-pain Service. Without the controller, this resource will not create a working load balancer.

### 4.3 Verify

```bash
kubectl get pods,svc -n id-pain
# Expect: id-pain, litellm, gong-http-server pods and services

kubectl logs -f deployment/id-pain -n id-pain
kubectl logs -f deployment/litellm -n id-pain
kubectl logs -f deployment/gong-http-server -n id-pain
```

---

### 4.4 Accessing the app

Pods must be **Running** (e.g. `1/1 READY`) before the app is reachable. If you see **Pending**, the pods are not scheduled yet (often due to not enough capacity on small nodes); check with `kubectl describe pod <pod-name> -n id-pain` and scale the node group or use larger instance types if needed.

**Option A – Port-forward (quick, from your machine)**

No public URL; traffic goes through your laptop. Useful for testing.

```bash
kubectl port-forward -n id-pain svc/id-pain 8080:80
```

Then open in a browser:

- **App UI:** http://localhost:8080  
- **Health:** http://localhost:8080/health  
- **API docs:** http://localhost:8080/docs  

Stop the forward with `Ctrl+C`.

**Option B – Public URL via Ingress (ALB)**

For a stable URL that others can use (e.g. for demos or sharing):

1. Install the [AWS Load Balancer Controller](https://kubernetes-sigs.github.io/aws-load-balancer-controller/v2.6/deploy/installation/) in your EKS cluster (Helm or manifest).
2. Apply the Ingress: `kubectl apply -f k8s/ingress.yaml`
3. Wait for the ALB to be created and get its DNS name:
   ```bash
   kubectl get ingress -n id-pain
   ```
4. Open the **ADDRESS** (e.g. `k8s-idpai-xxxxx.us-east-1.elb.amazonaws.com`) in a browser. HTTP only unless you add TLS (e.g. ACM + annotation).

---

## 5. Grafana Cloud monitoring (recommended)

Use [Grafana Cloud](https://grafana.com/products/cloud/) so you don’t run Prometheus/Grafana in the cluster. Metrics from your EKS cluster and id-pain are sent to Grafana Cloud and viewed in hosted Grafana.

### 5.1 Sign up and create a stack

1. Go to [grafana.com/products/cloud](https://grafana.com/products/cloud/) and sign up (free tier is enough for a demo).
2. Create a **stack** (or use the default). Note:
   - **Grafana URL** (e.g. `https://xxx.grafana.net`)
   - **Prometheus URL** and **Remote Write** endpoint (under Prometheus in the stack)
   - **User** and **API token** or **Remote Write password** for sending metrics.

You’ll use these when installing the Grafana Agent.

### 5.2 Install Grafana Agent in the cluster

The Grafana Agent scrapes Prometheus metrics (e.g. from the id-pain service or node/pod metrics) and forwards them to Grafana Cloud.

1. Add the Grafana Helm repo and install the agent (replace `YOUR_STACK`, `YOUR_REMOTE_WRITE_USER`, `YOUR_REMOTE_WRITE_PASSWORD` with values from your Grafana Cloud stack; get the remote write URL from the Prometheus section of the stack):

   ```bash
   helm repo add grafana https://grafana.github.io/helm-charts
   helm repo update
   helm install grafana-agent grafana/grafana-agent \
     --namespace grafana-agent \
     --create-namespace \
     -f - <<EOF
   config:
     metrics:
       global:
         remote_write:
           - url: https://prometheus-prod-XX-XXX.grafana.net/api/prom/push
             basic_auth:
               username: YOUR_REMOTE_WRITE_USER
               password: YOUR_REMOTE_WRITE_PASSWORD
       configs:
         - name: id-pain
           scrape_configs:
             - job_name: id-pain
               kubernetes_sd_configs:
                 - role: endpoints
                   namespaces:
                     names: [id-pain]
               relabel_configs:
                 - source_labels: [__meta_kubernetes_service_name]
                   action: keep
                   regex: id-pain
                 - source_labels: [__meta_kubernetes_endpoint_port_name]
                   action: keep
                   regex: http
   EOF
   ```

2. In the Grafana Cloud UI, add a **Prometheus** data source that uses the same Prometheus backend as your stack (usually pre-configured). Your dashboards will query this data source.

### 5.3 Expose /metrics from id-pain (optional but recommended)

For request rate, latency, and error rate in Grafana, expose Prometheus metrics from the FastAPI app:

1. In the id-pain repo: `pip install prometheus-client prometheus-fastapi-instrumentator`
2. In `main.py`, after `app = FastAPI(...)` add:
   ```python
   from prometheus_fastapi_instrumentator import Instrumentator
   Instrumentator().instrument(app).expose(app, endpoint="/metrics")
   ```
3. Rebuild and push the id-pain image, then redeploy. The Grafana Agent scrape config above will pick up the `id-pain` Service; ensure the app serves `/metrics` on the same port as the Service (8080).

### 5.4 Create a dashboard in Grafana Cloud

1. Log in to your Grafana Cloud Grafana (e.g. `https://xxx.grafana.net`).
2. Create a new dashboard (e.g. “id-pain – SA Call Analyzer”).
3. Add panels using the Prometheus data source, for example:
   - **Request rate:** `rate(http_requests_total{job="id-pain"}[5m])` or similar, depending on the instrumentator labels.
   - **Latency (e.g. p95):** `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="id-pain"}[5m]))`
   - **Pod CPU/memory:** use `container_*` or `node_*` metrics if the agent scrapes cAdvisor/kubelet.
4. Add a **Text** or **Link** panel with the URL to your Arize project so the demo narrative is “metrics in Grafana, traces in Arize.”

### 5.6 In-cluster option (Prometheus + Grafana)

If you prefer to run everything in the cluster instead of Grafana Cloud:

1. Install [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack) (includes Prometheus + Grafana).
2. Configure Prometheus to scrape the `id-pain` service (and add `/metrics` to the app as in 5.3).
3. Create the same style of dashboard and link to Arize.

### 5.7 Demo tips

- **Pre-build one dashboard** (e.g. “id-pain – SA Call Analyzer”) with 3–4 panels so the demo starts with a clear narrative.
- **Add a link to Arize** (link panel or text) to your Arize project so you can say: “Metrics here, full traces and LLM visibility in Arize.”
- **Generate light load** before the demo (e.g. hit `/health` or a simple `/api/` endpoint) so graphs aren’t empty.
- **Optional:** Add one alert (e.g. error rate or latency) and show it in the UI.

---

## 6. Summary checklist

| Step | Action |
|------|--------|
| 1 | Build and push **id-pain**, **litellm**, and **gong-http-server** images to ECR (Section 2) |
| 2 | Create or use an EKS cluster; update kubeconfig |
| 3 | Create secrets from examples: `secret.yaml`, `litellm-secret.yaml`, `gong-secret.yaml`; fill in and apply |
| 4 | Apply `k8s/` manifests in order: namespace → LiteLLM → Gong → id-pain (Section 4.2) |
| 5 | Access the app: port-forward (Section 4.4) or Ingress for a public URL |
| 6 | (Optional) Set up Grafana Cloud: install Grafana Agent, add `/metrics` to id-pain, create dashboard (Section 5) |
| 7 | To save cost: run `./scripts/down.sh` to delete the cluster; run `./scripts/up.sh` to recreate and deploy (Section 8) |

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

---

## 8. Bringing the cluster up and down

You can delete the EKS cluster when you’re not using it (to avoid paying for the control plane and nodes) and recreate it later. **Your “state” is preserved** because it lives outside the cluster.

### 8.1 What is preserved (nothing important is lost)

| What | Where it lives | When you delete the cluster |
|------|----------------|------------------------------|
| **Container images** | ECR (AWS) | Stay in ECR; no action needed |
| **Manifests** (k8s/*.yaml) | Git / your machine | Stay; re-apply when you bring the cluster back |
| **Secrets** (API keys, GCP base64) | Local files: `k8s/secret.yaml`, `k8s/litellm-secret.yaml`, `k8s/gong-secret.yaml` | Stay on your machine (they’re in .gitignore). Don’t delete these files |
| **Application data** | id-pain is stateless; BigQuery/Arize/Gong are external | No in-cluster data to lose |
| **Grafana Cloud** | Grafana’s servers | Unaffected; metrics and dashboards remain |

There are **no PersistentVolumes** in this setup, so deleting the cluster does not delete any on-disk data in the cluster itself.

### 8.2 Option A: Bring pods down (scale nodes to 0) — recommended for frequent on/off

Keeps the cluster and all k8s resources (Deployments, Services, Secrets, ConfigMaps). Only worker nodes are scaled to 0, so **no EC2 cost**; you still pay **~\$73/month** for the EKS control plane. Bringing pods back is **fast** (no cluster create, no re-apply).

**Pods down (stop running workloads):**

```bash
./scripts/down-pods.sh
```

This runs `eksctl scale nodegroup ... --nodes=0`. No worker nodes → no place for pods → your app is effectively off.

**Pods up (start workloads again):**

```bash
./scripts/up-pods.sh
```

This scales the node group back to 2 (override with `EKS_NODES=1` if you want). Nodes come up, the scheduler places your existing Deployments’ pods, and the app is back. **No need to re-apply manifests or push images.**

Override cluster/nodegroup/region:

```bash
EKS_CLUSTER_NAME=id-pain-demo EKS_NODEGROUP_NAME=standard-workers AWS_REGION=us-east-1 ./scripts/down-pods.sh
EKS_NODES=1 ./scripts/up-pods.sh
```

### 8.3 Option B: Delete the cluster (zero EKS cost)

When you want **no EKS cost at all** (e.g. not using the app for weeks), delete the cluster. You’ll recreate it and re-apply when you need it again.

**Bring down (delete cluster):**

```bash
./scripts/down.sh
```

This runs `eksctl delete cluster --name id-pain-demo --region us-east-1`. Cluster and nodes are removed. ECR images and your local secret YAML files are **not** deleted.

**Bring up (create cluster and deploy):**

**Prerequisites:** You’ve already pushed images to ECR (Section 2) and created the three secret files from the `.example` templates at least once.

```bash
./scripts/up.sh
```

This creates the cluster (~15–25 min), updates kubeconfig, and applies all k8s manifests. Then:

```bash
kubectl port-forward -n id-pain svc/id-pain 9090:80
# Open http://localhost:9090
```

Override cluster name or region: `EKS_CLUSTER_NAME=my-demo AWS_REGION=us-west-2 ./scripts/down.sh` (and same for `up.sh`).

### 8.4 If pods stay Pending (Insufficient memory / Too many pods)

If you see `0/X nodes are available: ... Insufficient memory, ... Too many pods`, the node group doesn’t have enough capacity. With **10 × t3.micro** you should have enough; if you scaled down and brought pods back with fewer nodes, scale back up to 10:

```bash
eksctl scale nodegroup \
  --cluster=id-pain-demo \
  --name=standard-workers \
  --nodes=10 \
  --region=us-east-1
```

Wait a few minutes for nodes to join; then the scheduler should place the pending pods:

```bash
kubectl get nodes
kubectl get pods -n id-pain
```

New clusters created with `./scripts/up.sh` use **10 × t3.micro** by default. When not in use, scale to 0 with `./scripts/down-pods.sh`; scale back to 10 with `./scripts/up-pods.sh`.

### 8.5 Summary: which option to use

| Goal | Scripts | Cost when “down” | Bring-back time |
|------|--------|-------------------|------------------|
| **Pods off, cluster stays** (frequent on/off) | `down-pods.sh` → `up-pods.sh` | ~\$73/month (control plane) | A few minutes (scale nodes up) |
| **Everything off** (long idle) | `down.sh` → `up.sh` | \$0 | ~15–25 min (create cluster + apply) |

---

## 9. Hypothesis research trace: where LLM inputs come from

**Full flow and diagrams:** See **[hypothesis_tool/TRACE_FLOW.md](hypothesis_tool/TRACE_FLOW.md)** for Mermaid flowcharts, data-flow diagram, and a table of every LLM call and where its input comes from (including **generate_hypotheses**).

When viewing a hypothesis research trace in Arize, LLM calls under **analyze_signals** get their input from the following places. Each relevant span includes a **metadata.input_sources** (or **metadata.llm_input_sources**) attribute that spells this out in the UI.

### 9.1 Company summary LLM (`company_summary_llm` / `company_summary_llm.invoke`)

- **Where the input is built:** `hypothesis_tool/analyzers/signal_extractor.py` → `SignalExtractor.analyze_with_llm()`.
- **What it uses:** Agent state keys **`ai_ml_results`**, **`job_results`**, **`news_results`** (each is a list of search results with title + description). Up to 10 AI/ML results, 5 job results, and 5 news results are formatted into a single prompt.
- **Where that state comes from:** The **execute_research** node (Brave search + job/news APIs) fills these keys before **analyze_signals** runs.

So: **execute_research** → state has `ai_ml_results`, `job_results`, `news_results` → **analyze_signals** runs → **company_summary_llm** builds the prompt from that state and calls the LLM.

### 9.2 Analyze_signals node (the span that “makes” the LLM call)

The **analyze_signals** node (LangGraph or `hypothesis_agent.analyze_signals`) runs **two** LLM calls:

1. **Company summary** (above): input from `ai_ml_results`, `job_results`, `news_results`.
2. **GenAI product extraction** (`extract_genai_product.llm_invoke`): input from **`genai_product_search_results`** (also populated by **execute_research**). The prompt is built in `research_agent._analyze_signals` from up to 5 results, 300 chars per description.

The **ChatAnthropic** span that appears under LangGraph’s **analyze_signals** in the tree is the **same** call as **company_summary_llm.invoke**; the instrumentor parents it to the LangGraph node. The **input** for that call is built in `SignalExtractor.analyze_with_llm` as above.

### 9.3 Quick reference

| Span | Input built from | Code location |
|------|------------------|----------------|
| **company_summary_llm.invoke** | `state.ai_ml_results`, `job_results`, `news_results` | `signal_extractor.analyze_with_llm()` |
| **extract_genai_product.llm_invoke** | `state.genai_product_search_results` | `research_agent._analyze_signals` |
| **extract_genai_product.llm_invoke_phase2** | Fetched page content from GenAI search URLs | `research_agent._analyze_signals` |

All of these state keys are populated by the **execute_research** node before **analyze_signals** runs.

---

## 10. Custom Demo Generator: traces not appearing in your Arize space

If you enter **Arize Space ID** and **API Key** in the Custom Demo Builder but traces don’t show up in that space, check the following.

### 10.1 Fixes applied in code

- **Env override:** The app now sets `ARIZE_SPACE_ID` and `ARIZE_API_KEY` from your form values for the duration of demo generation, then restores the previous values. That way the Arize OTLP exporter (and any SDK code that reads env) uses your credentials.
- **Stripping:** Space ID and API Key are trimmed of leading/trailing whitespace (e.g. pasted newlines).

### 10.2 What to verify

1. **Project name:** Traces are sent to a project named **`{project_name}-traces`** (e.g. `acme-demo-traces`). In Arize, open your **space** and check for a project with that suffix. The “Open in Arize” link at the end of the run uses that project.
2. **Delay:** Traces can take **30–60 seconds** to show up. Refresh the project page or wait a minute and try again.
3. **Credentials:** Use the **Space ID** from the space URL or from Arize → Settings → Space API Keys. Use an **API key** that has access to that space (from the same Settings page). No extra spaces or newlines when pasting.
4. **Response:** If the run finishes and shows “Traces sent to Arize project: …” and an Arize link, `traces_sent_to_arize` is true and the backend did create a demo provider with your credentials. If you see “Traces were not sent to Arize — no Space ID or API key”, the request didn’t include both fields (check the form and that the frontend sends `arize_space_id` and `arize_api_key` in the request body).

### 10.3 If it still fails

- Check server logs for `❌ Demo trace` or `❌ Demo pipeline` errors.
- Confirm the Arize API key is valid and not revoked (e.g. create a new key in Arize and try again).
- Ensure the Space ID is the correct base64-style ID for the space you’re opening in the UI.
