# Examples

## Guardrails with AWS Bedrock

[guardrails_bedrock.py](guardrails_bedrock.py) runs the same Guardrails use case (content safety, jailbreak, toxicity, PII checks) using a Claude model on **AWS Bedrock** instead of OpenAI or Anthropic direct.

### Setup

1. Install dependencies (includes `langchain-aws`):

   ```bash
   uv sync
   # or: pip install -e .
   # or: pip install langchain-aws
   ```

2. Configure AWS credentials (Bedrock access):

   - **Option A:** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optionally `AWS_REGION` (e.g. `us-east-1`)
   - **Option B:** `AWS_PROFILE` with a profile that has Bedrock access
   - **Option C:** Default AWS CLI profile (`aws configure`)

3. Enable the model in AWS Bedrock (e.g. Anthropic Claude 3.5 Sonnet) in the [Bedrock console](https://console.aws.amazon.com/bedrock/) → Model access.

### Run

```bash
# From repo root (default Bedrock model: Claude 3.5 Sonnet)
python examples/guardrails_bedrock.py

# Custom query
python examples/guardrails_bedrock.py --query "What is our refund policy?"

# Custom Bedrock model
BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0 python examples/guardrails_bedrock.py

# Send traces to Arize
ARIZE_SPACE_ID=... ARIZE_API_KEY=... python examples/guardrails_bedrock.py
```

### Using Bedrock in the Custom Demo Builder

You can run the **Guardrails** use case with Bedrock from the ID-PAIN Custom Demo Builder by using a model value that starts with `bedrock/`. If your deployment supports a custom model input, set the model to:

- `bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0`
- or `bedrock/<your-bedrock-model-id>`

The app uses `get_chat_llm()` in `arize_demo_traces.llm`: any model string with the `bedrock/` prefix is passed to LangChain’s `ChatBedrock` (requires `langchain-aws` and AWS credentials in the environment).
