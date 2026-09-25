"""Construcción del grafo: topología jerárquica Supervisor -> Especialistas.

El Supervisor es el único nodo con aristas condicionales; los especialistas
siempre vuelven a él (nunca se comunican directamente entre sí), lo cual
mantiene al Supervisor como única fuente de verdad sobre el estado del flujo.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.analyst_agent import analyst_node
from agents.research_agent import research_node
from agents.supervisor import supervisor_node
from state import OrchestratorState


def route_from_supervisor(state: OrchestratorState) -> str:
    next_agent = state["next_agent"]
    return END if next_agent == "FINISH" else next_agent


def build_graph():
    graph = StateGraph(OrchestratorState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", research_node)
    graph.add_node("analyst", analyst_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"researcher": "researcher", "analyst": "analyst", END: END},
    )
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")

    return graph.compile()


app = build_graph()
