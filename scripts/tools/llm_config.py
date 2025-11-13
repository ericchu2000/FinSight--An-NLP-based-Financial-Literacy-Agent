# scripts/tools/llm_config.py
"""
Centralized LLM configuration.

IMPORTANT:
- Commit this file with the placeholder key.
- On your local machine, replace the API_KEY with your real key.
- Optionally add this file to .gitignore if you don't want to risk committing the real key.
"""

OPENAI_API_KEY = "somethingsomething"

# You can change the model here without touching other files
OPENAI_MODEL = "gpt-4.1-nano"  # e.g. "gpt-4.1", "gpt-4.1-mini", etc.

# Base URL for OpenAI
OPENAI_BASE_URL = "https://api.openai.com/v1"
