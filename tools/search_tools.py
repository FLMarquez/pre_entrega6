"""Herramientas de búsqueda para el Agente de Investigación.

`buscar_pre_entregas` simula una consulta a una base vectorial: en lugar de
comparar embeddings, hace un scoring simple por superposición de palabras
clave sobre `data/pre_entregas_kb.json` (el "historial" de feedback de
pre-entregas anteriores del curso). Es intencionalmente simple para no
depender de una API externa ni de una base vectorial real, pero respeta la
misma interfaz que tendría un retriever real (query -> documentos rankeados).

Si se define `TAVILY_API_KEY` en el entorno, se agrega además una tool real
de búsqueda web con Tavily, para poder investigar temas que no estén en la
base local.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_core.tools import tool

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "pre_entregas_kb.json"


def _load_kb() -> list[dict]:
    with open(KB_PATH, encoding="utf-8") as f:
        return json.load(f)


@tool
def buscar_pre_entregas(query: str) -> str:
    """Busca en la base vectorial simulada de pre-entregas anteriores del curso.

    Devuelve, para la consulta dada, los documentos más relevantes (título,
    calificación y comentario textual) encontrados entre el feedback histórico
    de pre-entregas. Usar SIEMPRE esta herramienta antes de responder: no
    inventar calificaciones ni comentarios que no vengan de acá.
    """
    kb = _load_kb()
    query_terms = {t.strip(".,;:").lower() for t in query.split() if len(t) > 3}

    scored: list[tuple[int, dict]] = []
    for doc in kb:
        haystack = f"{doc['titulo']} {doc['comentario']} {doc['pre_entrega']}".lower()
        score = sum(1 for t in query_terms if t in haystack)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)

    top = [d for _, d in scored[:5]] or kb[:5]
    lines = [
        f"- [{d['pre_entrega']}] {d['titulo']} (calificación: {d['calificacion']}/10): "
        f"\"{d['comentario']}\""
        for d in top
    ]
    return "\n".join(lines)


def get_research_tools() -> list:
    """Devuelve la lista de tools del Agente de Investigación.

    Agrega Tavily solo si hay una API key configurada, para no romper el
    demo local cuando no se dispone de esa credencial.
    """
    tools = [buscar_pre_entregas]
    if os.getenv("TAVILY_API_KEY"):
        from langchain_tavily import TavilySearch

        tools.append(TavilySearch(max_results=3))
    return tools
