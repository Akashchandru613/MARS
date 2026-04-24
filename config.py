"""
MARS — Multi-Agent Research System
Central configuration loaded from environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS   = int(os.getenv("MAX_TOKENS", "4096"))
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.2"))

# ── Search ────────────────────────────────────────────────────────────────────
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
MAX_DOCS_PER_QUERY = int(os.getenv("MAX_DOCS_PER_QUERY", "3"))

# ── Tavily (optional, parallel web search source) ────────────────────────────
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
USE_TAVILY     = os.getenv("USE_TAVILY", "false").lower() in ("true", "1", "yes")

# ── Paths ─────────────────────────────────────────────────────────────────────
import pathlib
ROOT_DIR   = pathlib.Path(__file__).parent
DATA_DIR   = ROOT_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Groq client (singleton) ───────────────────────────────────────────────────
from groq import Groq

def get_groq_client() -> Groq:
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key. "
            "Get a free key at https://console.groq.com"
        )
    return Groq(api_key=GROQ_API_KEY)
