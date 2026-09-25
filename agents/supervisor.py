"""Supervisor: router inteligente y controlador de flujo.

Separa a propósito dos responsabilidades:

1. `decide_next_agent` (función pura, sin LLM): aplica las reglas "duras" que
   garantizan terminación. El LLM NUNCA puede saltarse estas reglas, así que
   por más que alucine una decisión rara, el grafo no puede quedar en bucle
   infinito. Esto es lo que evita el error común del "Supervisor Infinito".
2. `supervisor_node` (usa LLM): solo aporta el juicio cualitativo de "¿el
   análisis actual está lo bastante bien hecho o conviene refinarlo?" a través
   de salida estructurada (`SupervisorReview`). Ese juicio se combina con las
   reglas duras en `decide_next_agent`, nunca las reemplaza.

Rúbrica de suficiencia (`is_analysis_sufficient`): el análisis se considera
suficiente si menciona un promedio numérico Y una etiqueta de sentimiento
explícita. Es la validación que pide el enunciado ("rúbrica personalizada en
el prompt del Supervisor") antes de dar el END.
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import Literal

from config import MAX_REFINEMENTS, MAX_STEPS, get_llm
from state import OrchestratorState

SUPERVISOR_RUBRIC = """Sos el Supervisor de un equipo de dos especialistas: un
Agente de Investigación y un Agente de Análisis. Tu trabajo es evaluar si el
análisis actual (`analysis_data`) es de buena calidad, dado los datos crudos
que lo originaron (`research_data`).

Rúbrica de suficiencia:
- Debe incluir un promedio numérico concreto (no solo "las calificaciones son buenas").
- Debe incluir una etiqueta de sentimiento explícita (positivo/negativo/mixto/neutro).
- No debe contradecir los datos crudos de investigación.

Si el análisis cumple la rúbrica, `quiere_refinar` debe ser False. Si le falta
alguno de esos elementos o es vago, `quiere_refinar` debe ser True. Justificá
tu decisión en una oración en español."""


class SupervisorReview(BaseModel):
    quiere_refinar: bool = Field(
        description="True si el análisis actual es insuficiente y el Agente de Análisis debería refinarlo."
    )
    razon: str = Field(description="Justificación breve de la decisión, en español.")


_review_chain = None


def _get_review_chain():
    global _review_chain
    if _review_chain is None:
        _review_chain = get_llm().with_structured_output(SupervisorReview)
    return _review_chain


def is_analysis_sufficient(analysis_text: str) -> bool:
    """Chequeo determinístico de la rúbrica (promedio + etiqueta de sentimiento)."""
    tiene_promedio = bool(re.search(r"promedio[^0-9]{0,10}\d+(\.\d+)?", analysis_text.lower()))
    tiene_sentimiento = bool(
        re.search(r"\b(positivo|negativo|mixto|neutro)\b", analysis_text.lower())
    )
    return tiene_promedio and tiene_sentimiento


def decide_next_agent(
    has_research: bool,
    has_analysis: bool,
    analysis_sufficient: bool,
    refinements_used: int,
    steps: int,
    llm_wants_refine: bool,
) -> Literal["researcher", "analyst", "FINISH"]:
    """Regla dura de enrutamiento. Sin llamadas a LLM: 100% determinística y testeable.

    Garantiza terminación en como mucho `MAX_STEPS` pasos de especialistas, y
    permite como mucho `MAX_REFINEMENTS` vueltas de refinamiento del Analista,
    sin importar lo que "opine" el LLM.
    """
    if steps >= MAX_STEPS:
        return "FINISH"
    if not has_research:
        return "researcher"
    if not has_analysis:
        return "analyst"

    needs_refine = (not analysis_sufficient or llm_wants_refine) and (
        refinements_used < MAX_REFINEMENTS
    )
    if needs_refine:
        return "analyst"
    return "FINISH"


def supervisor_node(state: OrchestratorState) -> dict:
    steps = state.get("steps", 0)
    has_research = bool(state.get("research_data"))
    has_analysis = bool(state.get("analysis_data"))
    refinements_used = state.get("refinements_used", 0)

    llm_wants_refine = False
    reason = "Todavía no hay suficiente información para evaluar la calidad del análisis."
    sufficient = False

    if has_research and has_analysis and steps < MAX_STEPS:
        sufficient = is_analysis_sufficient(state["analysis_data"])
        context = (
            f"Consulta original del usuario:\n{state.get('user_query', '')}\n\n"
            f"Datos de investigación:\n{state['research_data']}\n\n"
            f"Análisis actual:\n{state['analysis_data']}\n\n"
            f"Chequeo automático de rúbrica (promedio + sentimiento explícitos): "
            f"{'CUMPLE' if sufficient else 'NO CUMPLE'}."
        )
        review = _get_review_chain().invoke(
            [SystemMessage(content=SUPERVISOR_RUBRIC), HumanMessage(content=context)]
        )
        llm_wants_refine = review.quiere_refinar
        reason = review.razon

    next_agent = decide_next_agent(
        has_research, has_analysis, sufficient, refinements_used, steps, llm_wants_refine
    )

    updates: dict = {
        "next_agent": next_agent,
        "task_completed": next_agent == "FINISH",
        "messages": [AIMessage(content=f"[Supervisor] {reason}", name="supervisor")],
    }
    if next_agent == "analyst" and has_analysis:
        updates["refinements_used"] = refinements_used + 1
    return updates
