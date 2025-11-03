"""Test LiteLLM connection with ChatOpenAI"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Test configuration
model_name = os.getenv("MODEL_NAME", "claude-3-5-haiku-20241022")
base_url = os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
if not base_url.endswith("/v1"):
    base_url = f"{base_url}/v1"

print(f"Testing LiteLLM connection...")
print(f"  Model: {model_name}")
print(f"  Base URL: {base_url}")
print()

try:
    llm = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=os.getenv("LITELLM_API_KEY", "dummy"),
        temperature=0.7
    )

    print("Sending test message...")
    response = llm.invoke("Say 'hello' in one word")
    print(f"✅ Success! Response: {response.content}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
