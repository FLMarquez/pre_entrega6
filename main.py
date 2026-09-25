"""Punto de entrada: ejecuta el orquestador con una consulta de usuario.

Uso:
    python main.py "¿Qué se dijo sobre la Pre-entrega 2 y qué sentimiento general hay?"

Si no se pasa una consulta por argumento, usa una consulta de demo que obliga
al Supervisor a delegar primero al Investigador y luego al Analista.
"""

from __future__ import annotations

import sys

from langchain_core.messages import HumanMessage

from graph import app

DEMO_QUERY = (
    "Investigá qué feedback recibieron las pre-entregas del curso relacionadas "
    "con RAG y extracción de entidades, y luego analizá el sentimiento general "
    "de los comentarios y el promedio de las calificaciones."
)


def run(query: str) -> dict:
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "next_agent": "researcher",
        "task_completed": False,
        "research_data": "",
        "analysis_data": "",
        "contributions": [],
        "steps": 0,
        "refinements_used": 0,
    }
    return app.invoke(initial_state)


def print_trace(final_state: dict) -> None:
    print("=" * 70)
    print("TRAZA DE CONTRIBUCIONES")
    print("=" * 70)
    for c in final_state["contributions"]:
        print(f"[paso {c['step']}] {c['agent']}: {c['summary']}")

    print("\n" + "=" * 70)
    print("RESPUESTA FINAL (Análisis)")
    print("=" * 70)
    print(final_state["analysis_data"])
    print(f"\ntask_completed = {final_state['task_completed']}")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or DEMO_QUERY
    print(f"Consulta: {query}\n")
    final_state = run(query)
    print_trace(final_state)
