"""
llm_client.py

Routes to either Ollama (local, free, used in development) or Groq
(free-tier hosted API, used when deployed since cloud hosts can't run
Ollama). Provider is chosen via LLM_PROVIDER in .env -- same pattern
as DATABASE_URL switching between SQLite and MySQL.
"""

import requests

from ml_detective.config.settings import settings
from ml_detective.utils.exceptions import MLDetectiveError
from ml_detective.utils.logger import get_logger

logger = get_logger(__name__)

_OLLAMA_URL = "http://localhost:11434/api/generate"
_OLLAMA_MODEL = "llama3.1:8b"

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "openai/gpt-oss-20b"

_TIMEOUT_SECONDS = 120


class LLMUnavailableError(MLDetectiveError):
    """Raised when the configured LLM provider isn't reachable."""


def generate_text(prompt: str) -> str:
    provider = settings.llm_provider.lower() or "ollama"

    if provider == "groq":
        return _generate_via_groq(prompt)
    return _generate_via_ollama(prompt)


def _generate_via_ollama(prompt: str) -> str:
    try:
        response = requests.post(
            _OLLAMA_URL,
            json={"model": _OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError as error:
        raise LLMUnavailableError("Could not connect to Ollama. Is it running?") from error
    except requests.exceptions.Timeout as error:
        raise LLMUnavailableError("Ollama request timed out.") from error


def _generate_via_groq(prompt: str) -> str:
    if not settings.groq_api_key:
        raise LLMUnavailableError("GROQ_API_KEY not set.")

    try:
        response = requests.post(
            _GROQ_URL,
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json={
                "model": _GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as error:
        raise LLMUnavailableError(f"Groq request failed: {error}") from error