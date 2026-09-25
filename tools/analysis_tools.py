"""Herramientas de cómputo para el Agente de Análisis.

Tres herramientas acotadas (cada una hace una sola cosa):
- `analizar_sentimiento`: análisis de sentimiento por léxico en español.
- `calcular_promedio`: estadística simple sobre las calificaciones encontradas.
- `validar_datos`: valida que el texto recibido tenga la forma mínima esperada
  (al menos una calificación numérica y un comentario), como paso de
  validación de esquema antes de darle el resultado al Supervisor.
"""

from __future__ import annotations

import re

from langchain_core.tools import tool

PALABRAS_POSITIVAS = {
    "excelente", "bueno", "buena", "buenos", "buenas", "genial", "claro",
    "clara", "prolijo", "prolija", "robusto", "robusta", "sólido", "solido",
    "sólida", "solida", "destacable", "correcto", "correcta", "completo",
    "completa", "notable", "mejora", "facilita", "consistente",
}
PALABRAS_NEGATIVAS = {
    "falta", "faltó", "falto", "faltaron", "débil", "debil", "confuso",
    "confusa", "incompleto", "incompleta", "error", "errores", "problema",
    "problemas", "insuficiente", "flojo", "floja", "recomienda", "corrupto",
    "corruptos", "difíciles", "dificiles", "pobres", "silenciosos",
}


@tool
def analizar_sentimiento(texto: str) -> str:
    """Analiza el sentimiento general (positivo/negativo/mixto/neutro) de un texto en español.

    Usa un conteo de palabras positivas vs. negativas sobre el texto recibido
    (por ejemplo, los comentarios devueltos por el Agente de Investigación) y
    devuelve una etiqueta explícita junto con el detalle del conteo.
    """
    palabras = re.findall(r"[a-záéíóúñ]+", texto.lower())
    positivas = sum(1 for w in palabras if w in PALABRAS_POSITIVAS)
    negativas = sum(1 for w in palabras if w in PALABRAS_NEGATIVAS)

    if positivas == 0 and negativas == 0:
        etiqueta = "neutro"
    elif positivas > negativas:
        etiqueta = "positivo"
    elif negativas > positivas:
        etiqueta = "negativo"
    else:
        etiqueta = "mixto"

    return (
        f"Sentimiento: {etiqueta} (señales positivas={positivas}, "
        f"señales negativas={negativas})"
    )


@tool
def calcular_promedio(numeros: str) -> str:
    """Calcula el promedio, mínimo y máximo de una lista de números encontrados en el texto.

    Extrae todos los números del texto recibido (por ejemplo, las
    calificaciones "8/10", "9/10" devueltas por el Agente de Investigación) y
    calcula estadísticas simples sobre ellos.
    """
    valores = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", numeros)]
    if not valores:
        return "No se encontraron números para calcular un promedio."

    promedio = sum(valores) / len(valores)
    return (
        f"Promedio: {promedio:.2f} (n={len(valores)}, "
        f"min={min(valores)}, max={max(valores)}, valores={valores})"
    )


@tool
def validar_datos(texto: str) -> str:
    """Valida que el texto tenga la forma mínima esperada de un dato de pre-entrega.

    Chequea que haya al menos un número (calificación) y una porción de texto
    libre (comentario) razonablemente larga. Sirve como paso de validación de
    esquema antes de entregarle el resultado final al Supervisor.
    """
    tiene_numero = bool(re.search(r"\d", texto))
    tiene_comentario = len(re.sub(r"[\d\W]", "", texto)) > 20

    if tiene_numero and tiene_comentario:
        return "Datos válidos: contienen al menos una calificación numérica y comentario textual."
    faltantes = []
    if not tiene_numero:
        faltantes.append("calificación numérica")
    if not tiene_comentario:
        faltantes.append("comentario textual")
    return f"Datos inválidos o insuficientes. Falta: {', '.join(faltantes)}."


def get_analysis_tools() -> list:
    """Devuelve la lista de tools del Agente de Análisis."""
    return [analizar_sentimiento, calcular_promedio, validar_datos]
