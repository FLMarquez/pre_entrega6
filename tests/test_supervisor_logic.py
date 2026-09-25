"""Tests de la lógica de enrutamiento del Supervisor.

No requieren API key: `decide_next_agent` e `is_analysis_sufficient` son
funciones puras (sin LLM), pensadas específicamente para poder verificar, sin
gastar cuota de ninguna API, que el grafo SIEMPRE termina y que las reglas de
refinamiento se respetan. Esto es lo que previene el "Supervisor Infinito".
"""

from agents.supervisor import decide_next_agent, is_analysis_sufficient
from config import MAX_REFINEMENTS, MAX_STEPS


def test_sin_research_va_a_researcher():
    assert (
        decide_next_agent(
            has_research=False,
            has_analysis=False,
            analysis_sufficient=False,
            refinements_used=0,
            steps=0,
            llm_wants_refine=False,
        )
        == "researcher"
    )


def test_con_research_sin_analysis_va_a_analyst():
    assert (
        decide_next_agent(
            has_research=True,
            has_analysis=False,
            analysis_sufficient=False,
            refinements_used=0,
            steps=1,
            llm_wants_refine=False,
        )
        == "analyst"
    )


def test_analisis_suficiente_y_llm_conforme_finaliza():
    assert (
        decide_next_agent(
            has_research=True,
            has_analysis=True,
            analysis_sufficient=True,
            refinements_used=0,
            steps=2,
            llm_wants_refine=False,
        )
        == "FINISH"
    )


def test_analisis_insuficiente_pide_refinamiento_una_vez():
    assert (
        decide_next_agent(
            has_research=True,
            has_analysis=True,
            analysis_sufficient=False,
            refinements_used=0,
            steps=2,
            llm_wants_refine=True,
        )
        == "analyst"
    )


def test_refinamiento_agotado_finaliza_aunque_llm_insista():
    """Aunque el LLM pida refinar de nuevo, el límite de refinamientos manda."""
    assert (
        decide_next_agent(
            has_research=True,
            has_analysis=True,
            analysis_sufficient=False,
            refinements_used=MAX_REFINEMENTS,
            steps=3,
            llm_wants_refine=True,
        )
        == "FINISH"
    )


def test_max_steps_fuerza_finish_sin_importar_nada():
    """Salvaguarda dura: al llegar a MAX_STEPS, se termina sí o sí."""
    assert (
        decide_next_agent(
            has_research=False,
            has_analysis=False,
            analysis_sufficient=False,
            refinements_used=0,
            steps=MAX_STEPS,
            llm_wants_refine=True,
        )
        == "FINISH"
    )


def test_es_terminacion_garantizada_en_como_mucho_max_steps_mas_refinamientos():
    """Simula muchos pasos consecutivos y verifica que nunca supera el techo esperado."""
    has_research = has_analysis = False
    sufficient = False
    refinements_used = 0
    steps = 0
    visited = []

    for _ in range(50):  # cota de seguridad del propio test, muy por encima de lo esperado
        next_agent = decide_next_agent(
            has_research, has_analysis, sufficient, refinements_used, steps, llm_wants_refine=True
        )
        visited.append(next_agent)
        if next_agent == "FINISH":
            break
        if next_agent == "researcher":
            has_research = True
        elif next_agent == "analyst":
            if has_analysis:
                refinements_used += 1
            has_analysis = True
            sufficient = False  # el LLM "insiste" en que nunca es suficiente
        steps += 1

    assert visited[-1] == "FINISH"
    assert steps <= MAX_STEPS


def test_is_analysis_sufficient_requiere_promedio_y_sentimiento():
    assert is_analysis_sufficient("El promedio es 7.5 y el sentimiento es positivo.")
    assert not is_analysis_sufficient("Las calificaciones son buenas en general.")
    assert not is_analysis_sufficient("El promedio es 7.5 pero no hay etiqueta clara.")
