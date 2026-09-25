"""Rotación automática de API keys de Gemini ante error de cuota (429).

Esto es una comodidad para desarrollo/demo (la capa gratuita de Google AI
Studio es muy acotada, p. ej. 20 requests/día para algunos modelos) y NO es
un requisito del enunciado de la pre-entrega: si tenés una sola key, `config.
get_llm()` ni siquiera importa este módulo.

`RotatingChatGoogleGenerativeAI` es un `ChatGoogleGenerativeAI` normal (hereda
toda su lógica de tool-calling) que, al recibir `GoogleRateLimitError`,
reconstruye su cliente interno con la siguiente API key de la lista y
reintenta, en vez de propagar el error. Si se agotan todas las keys, sí
propaga el error (no oculta un fallo real de cuota agotada por completo).
"""

from __future__ import annotations

from google.genai import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import GoogleRateLimitError
from pydantic import Field, SecretStr


class RotatingChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    api_keys: list[str] = Field(default_factory=list, exclude=True, repr=False)
    key_index: int = Field(default=0, exclude=True, repr=False)

    def _rotate_key(self) -> None:
        self.key_index += 1
        new_key = self.api_keys[self.key_index]
        self.google_api_key = SecretStr(new_key)
        self.client = Client(api_key=new_key)
        print(
            f"[rotating_llm] Cuota agotada, cambiando a GEMINI_API_KEY #{self.key_index + 1}/{len(self.api_keys)}."
        )

    def _generate(self, *args, **kwargs):
        while True:
            try:
                return super()._generate(*args, **kwargs)
            except GoogleRateLimitError:
                if self.key_index + 1 >= len(self.api_keys):
                    raise
                self._rotate_key()
