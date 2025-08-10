import os
import anthropic
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
solar_client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1/solar",
    default_headers={
        "Authorization": f"Bearer {UPSTAGE_API_KEY}"
    }
)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
CLAUDE_API_KEY = os.getenv('ANTHROPIC_API_KEY')
