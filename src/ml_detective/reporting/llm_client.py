"""
llm_client.py

Thin wrapper around Ollama's local HTTP API. No API key, no internet
needed after the model is pulled -- runs entirely on your machine.
"""

import requests

from ml_detective.config.settings import settings
from ml_detective.utils.exceptions import MLDetectiveError
from ml_detective.utils.logger import get_logger

logger = get_logger(__name__)

_OLLAMA_URL = "http://localhost:11434/api/generate"
_DEFAULT_MODEL = "llama3.1:8b"
_TIMEOUT_SECONDS = 120


class LLMUnavailableError(MLDetectiveError):
    """Raised when Ollama isn't running or the request fails."""


def generate_text(prompt: str, model: str = _DEFAULT_MODEL) -> str:
    try:
        response = requests.post(
            _OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError as error:
        raise LLMUnavailableError(
            "Could not connect to Ollama. Is it running? (try: ollama serve)"
        ) from error
    except requests.exceptions.Timeout as error:
        raise LLMUnavailableError("Ollama request timed out.") from error