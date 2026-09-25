"""Tests de las herramientas de investigación y análisis (sin LLM ni red)."""

from tools.analysis_tools import analizar_sentimiento, calcular_promedio, validar_datos
from tools.search_tools import buscar_pre_entregas


def test_buscar_pre_entregas_encuentra_por_palabra_clave():
    resultado = buscar_pre_entregas.invoke({"query": "RAG accidentes laborales embeddings"})
    assert "RAG" in resultado or "accidentes laborales" in resultado


def test_buscar_pre_entregas_sin_match_devuelve_algo():
    resultado = buscar_pre_entregas.invoke({"query": "xyz inexistente"})
    assert resultado  # fallback a los primeros documentos, nunca vacío


def test_calcular_promedio_extrae_numeros():
    resultado = calcular_promedio.invoke({"numeros": "calificaciones: 8, 9 y 6"})
    assert "Promedio: 7.67" in resultado


def test_calcular_promedio_sin_numeros():
    resultado = calcular_promedio.invoke({"numeros": "no hay nada acá"})
    assert "No se encontraron números" in resultado


def test_analizar_sentimiento_detecta_positivo():
    resultado = analizar_sentimiento.invoke(
        {"texto": "Excelente y prolijo, muy sólido y consistente"}
    )
    assert "positivo" in resultado


def test_analizar_sentimiento_detecta_negativo():
    resultado = analizar_sentimiento.invoke(
        {"texto": "Faltaron validaciones, hubo errores y problemas insuficientes"}
    )
    assert "negativo" in resultado


def test_validar_datos_ok():
    resultado = validar_datos.invoke(
        {"texto": "Calificación 8/10, comentario extenso sobre la entrega del proyecto"}
    )
    assert "válidos" in resultado


def test_validar_datos_insuficientes():
    resultado = validar_datos.invoke({"texto": "corto"})
    assert "inválidos" in resultado or "insuficientes" in resultado
