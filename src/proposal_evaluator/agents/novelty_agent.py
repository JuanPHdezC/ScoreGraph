import json
import structlog
from typing import Any
from proposal_evaluator.config.config_repository import get_rubric_by_criterio
from proposal_evaluator.llm.client import call_structured, LLMCallError
from proposal_evaluator.schemas import (
    EvaluationState,
    NoveltyEvaluation,
    Criterio,
)
from proposal_evaluator.agents.completeness_gate import check_completeness_from_rubric

logger = structlog.get_logger(__name__)

# ─── MOCK PRIOR ART SEARCH TOOL ───
# En producción, esta función se conectaría a:
# - Bases de patentes reales (USPTO, EPO, WIPO via APIs)
# - Histórico de propuestas de Wazoku (vector search semántico)
# - Literatura técnica (Google Scholar, arXiv, etc.)
# Para el MVP, simula resultados deterministas basados en palabras clave.

MOCK_PATENT_DB = {
    "blockchain": [
        {"id": "US10123456B2", "title": "Blockchain-based supply chain tracking", "relevance": 0.85},
        {"id": "EP3456789A1", "title": "Distributed ledger for asset management", "relevance": 0.72},
    ],
    "ia generativa": [
        {"id": "US11223344B2", "title": "Generative AI for content creation", "relevance": 0.91},
        {"id": "WO2023123456A1", "title": "Large language model fine-tuning", "relevance": 0.68},
    ],
    "iot": [
        {"id": "US10987654B2", "title": "IoT sensor network optimization", "relevance": 0.78},
    ],
    "computación cuántica": [
        {"id": "US11556677B2", "title": "Quantum error correction method", "relevance": 0.82},
    ],
}


def mock_prior_art_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Simula búsqueda de prior art / patentes relacionadas.
    En producción: replace with real patent DB API + Wazoku historical proposals search.

    Args:
        query: Términos de búsqueda (ej. descripción de la innovación)
        max_results: Máximo resultados a devolver

    Returns:
        Lista de dicts con id, title, relevance (0-1)
    """
    query_lower = query.lower()
    results = []
    for keyword, patents in MOCK_PATENT_DB.items():
        if keyword in query_lower:
            results.extend(patents)
    # Ordenar por relevance desc y limitar
    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results[:max_results]


def detect_patent_collision(search_results: list[dict], threshold: float = 0.8) -> bool:
    """
    Determina si hay colisión de patente basada en relevance.
    En producción: análisis legal más sofisticado.
    """
    return any(r["relevance"] >= threshold for r in search_results)


# ─── AGENTE NOVELTY ───
SYSTEM_PROMPT_TEMPLATE = """Eres un evaluador experto en novedad e innovación de propuestas tecnológicas.

Tu tarea: Evaluar el grado de novedad de la propuesta. PUEDES INVOCAR la herramienta
`prior_art_search` para buscar patentes/arte previo relacionado (máx 2 iteraciones).

CRITERIOS DE EVALUACIÓN:
{criterios_de_evaluacion}

ESCALA DE REFERENCIA (usa como guía, no como regla estricta):
{escala_guia}

REGLAS ESTRICTAS:
- Devuelve SOLO el JSON estructurado según el schema.
- score: entero 0-100.
- summary: razonamiento interno breve (máx 500 chars), NO para usuario final.
- confidence_fields_present: true si la info + búsqueda permiten evaluar con fundamento.
- colision_patente_detectada: true si la búsqueda revela riesgo de infracción (relevance >= 0.8).
- fuentes_consultadas: lista de fuentes/IDs consultados (ej. ["USPTO:US10123456B2", "EPO:EP3456789A1"]).

FLUJO RECOMENDADO:
1. Analiza la propuesta y extrae términos clave de innovación.
2. Invoca `prior_art_search` con esos términos.
3. Si los resultados son insuficientes, refina la búsqueda (máx 1 refinamiento = 2 iteraciones totales).
4. Determina colisión y novedad, y responde con el JSON final.
"""

PRIOR_ART_TOOL = {
    "name": "prior_art_search",
    "description": "Busca patentes y arte previo relacionado con términos de innovación",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Términos de búsqueda (palabras clave de la innovación)"},
            "max_results": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    },
}


def build_system_prompt(rubric) -> str:
    escala_lines = [f"- {rango}: {desc}" for rango, desc in rubric.escala_guia.items()]
    return SYSTEM_PROMPT_TEMPLATE.format(
        criterios_de_evaluacion=rubric.criterios_de_evaluacion,
        escala_guia="\n".join(escala_lines),
    )


def build_user_message(state: EvaluationState) -> str:
    parts = [
        f"PROPUESTA (texto libre):\n{state.proposal_text}\n",
    ]
    if state.raw_proposal_data:
        parts.append("DATOS ESTRUCTURADOS PROVISTOS:")
        for k, v in state.raw_proposal_data.items():
            parts.append(f"  {k}: {v}")
    parts.append("\nEvalúa la NOVEDAD. Usa la herramienta prior_art_search si lo necesitas.")
    return "\n".join(parts)


def novelty_agent(state: EvaluationState) -> EvaluationState:
    """
    Agente evaluador de Novedad con tool-calling simulado de prior art search.
    - Obtiene rúbrica, verifica completitud.
    - Si hay info suficiente, ejecuta ciclo LLM + tool (máx 2 iteraciones tool).
    - Actualiza state.novelty_result y missing_fields_by_agent.
    """
    proposal_id = state.proposal_id
    logger.info("agent_start", agent="novelty", proposal_id=proposal_id)

    rubric = get_rubric_by_criterio(Criterio.NOVELTY)
    if not rubric:
        logger.error("rubric_not_found", criterio="novelty")
        state.missing_fields_by_agent["novelty"] = ["rubric_not_configured"]
        return state

    missing = check_completeness_from_rubric(proposal_id, rubric, state.raw_proposal_data)
    if missing:
        state.missing_fields_by_agent["novelty"] = missing
        logger.info("agent_blocked_missing_fields", agent="novelty", missing=missing)
        return state

    try:
        system_prompt = build_system_prompt(rubric)
        user_message = build_user_message(state)

        # Ciclo con tool-calling (máx 2 iteraciones de tool)
        messages = [{"role": "user", "content": user_message}]
        tool_iterations = 0
        max_tool_iterations = 2
        fuentes_consultadas = []

        while tool_iterations < max_tool_iterations:
            response = call_structured_with_tool(
                system_prompt, messages, NoveltyEvaluation, PRIOR_ART_TOOL
            )

            # Verificar si hay tool_use
            tool_use = getattr(response, "tool_use", None) if hasattr(response, "tool_use") else None
            # En esta implementación simplificada, call_structured no expone tool_use directamente.
            # Para MVP, asumimos que el LLM responde directamente con el JSON final.
            # NOTA: En implementación completa, se usaría client.messages.create con tool_choice
            # y se manejaría el bucle tool_use → tool_result aquí.

            if isinstance(response, NoveltyEvaluation):
                # LLM devolvió resultado final
                response.fuentes_consultadas = fuentes_consultadas
                state.novelty_result = response
                state.missing_fields_by_agent["novelty"] = []
                logger.info(
                    "agent_completed",
                    agent="novelty",
                    score=response.score,
                    colision=response.colision_patente_detectada,
                    fuentes=fuentes_consultadas,
                )
                return state

            tool_iterations += 1

        # Si se agotan iteraciones sin resultado final, error
        raise LLMCallError("Máximo de iteraciones de tool alcanzado sin resultado final")

    except LLMCallError as e:
        logger.error("agent_llm_failed", agent="novelty", error=str(e))
        state.missing_fields_by_agent["novelty"] = ["llm_error"]

    return state


# Helper simplificado para MVP: usa call_structured sin tool loop real
# (la tool se documenta en el prompt pero el LLM responde directo)
def call_structured_with_tool(system_prompt, messages, output_schema, tool_def):
    """
    Wrapper que para MVP llama call_structured directo.
    En producción completa, aquí iría el loop de tool-calling real.
    """
    # Construir user message combinando todo
    combined_user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
    return call_structured(system_prompt, combined_user, output_schema)