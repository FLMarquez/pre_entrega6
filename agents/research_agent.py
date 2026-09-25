"""Agente de Investigación: busca información, no la interpreta.

Se construye con `create_react_agent` (LangGraph) y una única tool acotada
(`buscar_pre_entregas`, + Tavily si hay API key). El LLM se instancia de
forma perezosa (`_get_agent`) para que importar este módulo no falle si
todavía no hay credenciales configuradas (por ejemplo, al generar el diagrama
del grafo con `graph.py` sin haber seteado `.env`).

Para evitar la "contaminación de contexto": el nodo NO le pasa al agente todo
`state["messages"]` (que incluye mensajes del Supervisor y del Analista).
Le pasa únicamente la consulta del usuario, en un `HumanMessage` nuevo.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from agents.utils import extract_text
from config import get_llm
from state import OrchestratorState
from tools.search_tools import get_research_tools

RESEARCH_SYSTEM_PROMPT = """Sos el Agente de Investigación de un equipo de análisis.

Tu única tarea es usar la herramienta `buscar_pre_entregas` para encontrar
información relevante a la consulta del usuario y devolver un resumen conciso
de los hallazgos, citando las calificaciones y comentarios textuales
encontrados. No inventes datos que no vengan de la herramienta. No analices
sentimiento ni hagas cálculos: eso es trabajo del Agente de Análisis, que
recibirá tu resumen a continuación."""

_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = create_react_agent(
            get_llm(), tools=get_research_tools(), prompt=RESEARCH_SYSTEM_PROMPT
        )
    return _agent


def research_node(state: OrchestratorState) -> dict:
    task = state.get("user_query", "")
    result = _get_agent().invoke({"messages": [HumanMessage(content=task)]})
    final_text = extract_text(result["messages"][-1])

    step = state.get("steps", 0) + 1
    return {
        "research_data": final_text,
        "messages": [AIMessage(content=final_text, name="researcher")],
        "contributions": [
            {"agent": "researcher", "summary": final_text[:200], "step": step}
        ],
        "steps": step,
    }
