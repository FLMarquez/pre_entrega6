"""Configuración compartida: selección de proveedor de LLM.

Sigue la misma convención que otras pre-entregas del curso (`.env` +
`get_llm()` centralizado): por defecto usa Gemini (capa gratuita de Google AI
Studio), pero se puede cambiar de proveedor con la variable de entorno
`LLM_PROVIDER` sin tocar el código de los agentes.

Proveedores soportados: gemini (default), openai, anthropic.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
MAX_STEPS = int(os.getenv("MAX_STEPS", "6"))
MAX_REFINEMENTS = int(os.getenv("MAX_REFINEMENTS", "1"))


def _collect_gemini_keys() -> list[str]:
    """Junta GOOGLE_API_KEY/GEMINI_API_KEY + GEMINI_API_KEY_1..N del .env.

    Soporta múltiples keys (por ejemplo, de distintos proyectos de AI Studio)
    para poder rotar entre ellas si una se queda sin cuota gratuita. Con una
    sola key definida, el comportamiento es idéntico al de antes.
    """
    keys: list[str] = []
    primary = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if primary:
        keys.append(primary)

    i = 1
    while True:
        extra = os.getenv(f"GEMINI_API_KEY_{i}")
        if not extra:
            break
        if extra not in keys:
            keys.append(extra)
        i += 1

    return keys


def get_llm(temperature: float = 0.0):
    """Devuelve el chat model configurado según `LLM_PROVIDER`.

    Se centraliza acá (en vez de instanciarlo en cada agente) para que todo
    el orquestador use el mismo modelo y sea trivial cambiar de proveedor.
    """
    if LLM_PROVIDER == "gemini":
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        keys = _collect_gemini_keys()
        if not keys:
            raise RuntimeError(
                "Falta GOOGLE_API_KEY. Copiá .env.example a .env y completá tu "
                "API key gratuita de Google AI Studio (https://aistudio.google.com/app/apikey)."
            )

        if len(keys) == 1:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model, google_api_key=keys[0], temperature=temperature
            )

        from rotating_llm import RotatingChatGoogleGenerativeAI

        return RotatingChatGoogleGenerativeAI(
            model=model,
            google_api_key=keys[0],
            temperature=temperature,
            api_keys=keys,
        )

    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Falta OPENAI_API_KEY en el .env.")
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature
        )

    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("Falta ANTHROPIC_API_KEY en el .env.")
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
            temperature=temperature,
        )

    raise ValueError(
        f"LLM_PROVIDER='{LLM_PROVIDER}' no soportado. Usá gemini, openai o anthropic."
    )
