"""Estado compartido del orquestador multi-agente.

Extiende `MessagesState` (que ya trae `messages` con el reducer `add_messages`)
con los campos necesarios para que el Supervisor pueda enrutar el flujo sin
perder contexto entre agentes:

- `contributions` registra QUÉ agente aportó QUÉ información y en qué paso,
  para poder auditar la conversación sin tener que re-parsear los mensajes.
- `research_data` / `analysis_data` son los "resultados" concretos de cada
  especialista (separados de `messages`, que es el log conversacional).
- `steps` y `refinements_used` son los contadores que usa el Supervisor para
  evitar el bucle infinito ("Supervisor Infinito"): ver `agents/supervisor.py`.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import MessagesState


class Contribution(TypedDict):
    """Un aporte puntual de un agente al estado compartido."""

    agent: str
    summary: str
    step: int


class OrchestratorState(MessagesState):
    """Estado compartido entre Supervisor, Investigador y Analista."""

    # Consulta original del usuario. Se guarda aparte de `messages` para que
    # cada especialista pueda recibir SOLO esto (y no todo el historial) al
    # ser invocado, evitando la "contaminación de contexto".
    user_query: str

    # Próximo nodo que debe ejecutarse, decidido por el Supervisor.
    next_agent: Literal["researcher", "analyst", "FINISH"]

    # True cuando el Supervisor considera que la tarea está lista para cerrarse.
    task_completed: bool

    # Resultado del Agente de Investigación (texto con hallazgos crudos).
    research_data: str

    # Resultado del Agente de Análisis (texto con métricas + sentimiento).
    analysis_data: str

    # Traza de quién contribuyó qué. `operator.add` concatena las listas que
    # devuelve cada nodo en vez de sobreescribir el estado.
    contributions: Annotated[list[Contribution], operator.add]

    # Contador global de pasos de especialistas ejecutados (no de supervisor).
    steps: int

    # Cuántas veces se le pidió al Analista refinar su output. Acotado por
    # MAX_REFINEMENTS en agents/supervisor.py.
    refinements_used: int
