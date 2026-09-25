"""Agente de Análisis: procesa los datos que ya trajo el Investigador.

No vuelve a buscar información: recibe `state["research_data"]` como único
contexto (más una nota de refinamiento si el Supervisor lo pidió) y usa sus
tools de cómputo (sentimiento, promedio, validación de esquema) para producir
una respuesta estructurada.

Igual que en `research_agent.py`, el agente se construye de forma perezosa y
el nodo evita pasarle el historial completo de `messages`: solo le pasa el
dato crudo que necesita procesar.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from config import get_llm
from state import OrchestratorState
from tools.analysis_tools import get_analysis_tools

ANALYST_SYSTEM_PROMPT = """Sos el Agente de Análisis de un equipo de investigación.

Recibís datos crudos ya recolectados por el Agente de Investigación
(calificaciones y comentarios de pre-entregas anteriores del curso). Tu tarea,
en este orden:
1. Usar `calcular_promedio` sobre las calificaciones encontradas en el texto.
2. Usar `analizar_sentimiento` sobre los comentarios encontrados en el texto.
3. Usar `validar_datos` para confirmar que la información de entrada era
   suficiente.

Devolvé un resumen breve y estructurado con: el promedio calculado, la
etiqueta de sentimiento general (positivo/negativo/mixto/neutro) y el
resultado de la validación. No vuelvas a buscar información nueva: trabajá
exclusivamente con el texto que se te pasó."""

_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = create_react_agent(
            get_llm(), tools=get_analysis_tools(), prompt=ANALYST_SYSTEM_PROMPT
        )
    return _agent


def analyst_node(state: OrchestratorState) -> dict:
    research_data = state.get("research_data", "")
    task = f"Datos a analizar:\n{research_data}"

    if state.get("refinements_used", 0) > 0:
        task += (
            "\n\nNota del Supervisor: tu análisis anterior fue considerado "
            "insuficiente. Asegurate de incluir un promedio numérico EXPLÍCITO "
            "y una etiqueta de sentimiento EXPLÍCITA (positivo/negativo/mixto/"
            "neutro) en tu respuesta final."
        )

    result = _get_agent().invoke({"messages": [HumanMessage(content=task)]})
    final_text = result["messages"][-1].content

    step = state.get("steps", 0) + 1
    return {
        "analysis_data": final_text,
        "messages": [AIMessage(content=final_text, name="analyst")],
        "contributions": [
            {"agent": "analyst", "summary": final_text[:200], "step": step}
        ],
        "steps": step,
    }
