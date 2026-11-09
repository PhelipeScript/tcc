"""Provedores de LLM usados na geração (mesmos do corpus inicial).

Todos falam HTTP. Groq, Mistral, NVIDIA e OpenRouter usam o formato OpenAI
`/chat/completions`. O LM Studio, local e da rede (lmstudio, qwen, glm), usa o
endpoint nativo `/api/v1/chat`, que permite desligar o raciocínio.

Chaves e endereços vêm do `.env` na raiz do projeto (veja `.env.example`).
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import requests
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

_SYSTEM = ("Você gera frases em português brasileiro para um corpus de pesquisa. "
           "Responda APENAS com um array JSON de strings, sem explicação e sem raciocínio.")


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Como falar com um provedor."""

    key: str
    kind: Literal["openai", "lmstudio_native"]
    url: str
    model_env: str
    default_model: str
    url_env: str | None = None
    api_key_env: str | None = None
    reasoning: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_payload: dict[str, Any] = field(default_factory=dict)


PROVIDERS: dict[str, ProviderSpec] = {
    # Groq (plano gratuito: ~8 mil tokens/min e 1.000 requisições/dia por modelo). Duas chaves, dois modelos.
    "groq": ProviderSpec("groq", "openai", "https://api.groq.com/openai/v1/chat/completions",
                         "GROQ_MODEL", "openai/gpt-oss-120b", api_key_env="GROQ_API_KEY",
                         extra_payload={"reasoning_effort": "low", "include_reasoning": False,
                                        "max_completion_tokens": 4096}),
    "groq2": ProviderSpec("groq2", "openai", "https://api.groq.com/openai/v1/chat/completions",
                          "GROQ_MODEL_2", "qwen/qwen3.8-27b", api_key_env="GROQ_API_KEY_2",
                          extra_payload={"reasoning_effort": "none", "max_completion_tokens": 3072}),
    # Chaves cruzadas (cada chave é de uma organização diferente; a cota diária é por modelo e organização).
    "groq_qwen": ProviderSpec("groq_qwen", "openai", "https://api.groq.com/openai/v1/chat/completions",
                              "GROQ_QWEN_MODEL", "qwen/qwen3.8-27b", api_key_env="GROQ_API_KEY",
                              extra_payload={"reasoning_effort": "none", "max_completion_tokens": 3072}),
    "groq2_gptoss": ProviderSpec("groq2_gptoss", "openai", "https://api.groq.com/openai/v1/chat/completions",
                                 "GROQ2_GPTOSS_MODEL", "openai/gpt-oss-120b", api_key_env="GROQ_API_KEY_2",
                                 extra_payload={"reasoning_effort": "low", "include_reasoning": False,
                                                "max_completion_tokens": 4096}),
    # Google AI Studio pelo endpoint compatível com OpenAI. gemini-3.5-flash aceita raciocínio "none";
    # o 3.8-flash não (levava ~100 s por lote).
    "gemini": ProviderSpec("gemini", "openai",
                           "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                           "GEMINI_MODEL", "gemini-3.5-flash", api_key_env="GEMINI_API_KEY",
                           extra_payload={"reasoning_effort": "none"}),
    "mistral": ProviderSpec("mistral", "openai", "https://api.mistral.ai/v1/chat/completions",
                            "MISTRAL_MODEL", "mistral-small-latest", api_key_env="MISTRAL_API_KEY"),
    "nvidia": ProviderSpec("nvidia", "openai", "https://integrate.api.nvidia.com/v1/chat/completions",
                           "NVIDIA_MODEL", "moonshotai/kimi-k3", api_key_env="NVIDIA_API_KEY",
                           extra_payload={"max_tokens": 2048}),
    # OpenRouter gratuito: 50 requisições por dia nos modelos ":free".
    "openrouter": ProviderSpec("openrouter", "openai", "https://openrouter.ai/api/v1/chat/completions",
                               "OPENROUTER_MODEL", "google/gemma-4-31b-it:free", api_key_env="OPENROUTER_API_KEY",
                               extra_headers={"X-Title": "TCC2-DDSD"},
                               extra_payload={"max_tokens": 2048}),
    "lmstudio": ProviderSpec("lmstudio", "lmstudio_native", "http://127.0.0.1:1234/api/v1/chat",
                             "LMSTUDIO_MODEL", "google/gemma-4-e4b", url_env="LMSTUDIO_URL",
                             reasoning="off"),
    "qwen": ProviderSpec("qwen", "lmstudio_native", "", "QWEN_MODEL", "qwen/qwen3.6-35b-a3b",
                         url_env="QWEN_BASE_URL", reasoning="off"),
    "glm": ProviderSpec("glm", "lmstudio_native", "", "GLM_MODEL", "zai-org/glm-4.6v-flash",
                        url_env="GLM_BASE_URL", reasoning="on"),
}


class ProviderError(RuntimeError):
    """Resposta inválida ou configuração ausente do provedor."""


class RateLimited(ProviderError):
    """Limite por minuto atingido; nova tentativa depois da espera indicada."""


class QuotaExhausted(ProviderError):
    """Cota diária esgotada: não adianta tentar de novo hoje (a geração para e pode ser retomada)."""


#: Mensagens de 429 que indicam cota DIÁRIA (Groq "requests per day (RPD)", Gemini "PerDay",
#: OpenRouter "free-models-per-day").
_DAILY_QUOTA = re.compile(r"per[ _-]?day|\bRPD\b|daily", re.IGNORECASE)


def _retry_after(response: requests.Response) -> float:
    """Espera sugerida pelo servidor (cabeçalho ou texto "try again in 2.5s"), limitada a 120 s."""
    header = response.headers.get("retry-after")
    if header:
        try:
            return min(float(header), 120.0)
        except ValueError:
            pass
    match = re.search(r"(?:try again in|retry in)\s*([0-9.]+)\s*(ms|s)", response.text, re.IGNORECASE)
    if match:
        value = float(match.group(1)) / (1000.0 if match.group(2).lower() == "ms" else 1.0)
        return min(value + 0.5, 120.0)
    return 15.0


class LLMClient:
    """Cliente HTTP de um provedor, com novas tentativas e backoff exponencial."""

    def __init__(self, spec: ProviderSpec) -> None:
        self.spec = spec
        self.model = os.environ.get(spec.model_env) or spec.default_model
        self.url = (os.environ.get(spec.url_env) if spec.url_env else None) or spec.url
        if not self.url:
            raise ProviderError(f"{spec.url_env} não definido no .env (endereço do servidor {spec.key})")
        self._api_key = os.environ.get(spec.api_key_env) if spec.api_key_env else None
        if spec.api_key_env and not self._api_key:
            raise ProviderError(f"{spec.api_key_env} não definida no .env")

    @retry(wait=wait_exponential(multiplier=2, min=2, max=60), stop=stop_after_attempt(12), reraise=True,
           retry=retry_if_not_exception_type(QuotaExhausted))
    def complete(self, prompt: str, temperature: float) -> str:
        """Texto da resposta do modelo para `prompt`."""
        headers = {"Content-Type": "application/json", **self.spec.extra_headers}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self.spec.kind == "openai":
            payload = {"model": self.model, "temperature": temperature,
                       "messages": [{"role": "system", "content": _SYSTEM},
                                    {"role": "user", "content": prompt}],
                       **self.spec.extra_payload}
        else:
            payload = {"model": self.model, "reasoning": self.spec.reasoning, "input": prompt,
                       "temperature": temperature}
        response = requests.post(self.url, headers=headers, json=payload, timeout=900)
        if response.status_code == 429:
            if _DAILY_QUOTA.search(response.text):
                raise QuotaExhausted(f"{self.spec.key}: cota diária esgotada ({response.text[:200]})")
            time.sleep(_retry_after(response))
            raise RateLimited(f"{self.spec.key}: limite por minuto")
        if response.status_code != 200:
            raise ProviderError(f"{self.spec.key} HTTP {response.status_code}: {response.text[:300]}")
        data = response.json()
        if self.spec.kind == "openai":
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
        else:
            # Endpoint nativo: modelos de raciocínio devolvem blocos; a resposta é o bloco "message".
            blocks = [b for b in data.get("output") or [] if isinstance(b, dict) and b.get("content")]
            messages = [b for b in blocks if b.get("type") in (None, "message")]
            content = (messages or blocks or [{}])[0].get("content")
        if not content or not str(content).strip():
            raise ProviderError(f"{self.spec.key} devolveu conteúdo vazio")
        return str(content)
