import os
from dotenv import load_dotenv

load_dotenv()

print(f'USE_LITELLM={os.getenv("USE_LITELLM")}')
print(f'MODEL_NAME={os.getenv("MODEL_NAME")}')
print(f'LITELLM_BASE_URL={os.getenv("LITELLM_BASE_URL")}')
print(f'ANTHROPIC_API_KEY={os.getenv("ANTHROPIC_API_KEY")[:20]}...' if os.getenv("ANTHROPIC_API_KEY") else 'ANTHROPIC_API_KEY=None')
