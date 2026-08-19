import structlog
from proposal_evaluator.config.config_repository import get_rubric_by_criterio
from proposal_evaluator.llm.client import call_structured, LLMCallError
from proposal_evaluator.schemas import (
    EvaluationState,
    ImpactEvaluation,
    Criterio,
)
from proposal_evaluator.agents.completeness_gate import check_completeness_from_rubric

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT_TEMPLATE = """Eres un evaluador experto en impacto de negocio y usuarios de propuestas de innovación.

Tu tarea: Evaluar el impacto potencial de la propuesta basándote ÚNICAMENTE en la información provista.

CRITERIOS DE EVALUACIÓN:
{criterios_de_evaluacion}

ESCALA DE REFERENCIA (usa como guía, no como regla estricta):
{escala_guia}

REGLAS ESTRICTAS:
- Devuelve SOLO el JSON estructurado según el schema.
- score: entero 0-100.
- summary: razonamiento interno breve (máx 500 chars), NO para usuario final.
- confidence_fields_present: true si la info provista permite evaluar con fundamento.

NO inventes datos. Si la info es insuficiente, el gate de completitud ya te habría bloqueado.
"""

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
    parts.append("\nEvalúa el IMPACTO y devuelve el JSON estructurado.")
    return "\n".join(parts)


def impact_agent(state: EvaluationState) -> EvaluationState:
    """
    Agente evaluador de Impacto.
    - Obtiene rúbrica, verifica completitud, llama LLM si hay info suficiente.
    - Actualiza state.impact_result y missing_fields_by_agent.
    """
    proposal_id = state.proposal_id
    logger.info("agent_start", agent="impact", proposal_id=proposal_id)

    rubric = get_rubric_by_criterio(Criterio.IMPACT)
    if not rubric:
        logger.error("rubric_not_found", criterio="impact")
        state.missing_fields_by_agent["impact"] = ["rubric_not_configured"]
        return state

    missing = check_completeness_from_rubric(proposal_id, rubric, state.raw_proposal_data)
    if missing:
        state.missing_fields_by_agent["impact"] = missing
        logger.info("agent_blocked_missing_fields", agent="impact", missing=missing)
        return state

    try:
        system_prompt = build_system_prompt(rubric)
        user_message = build_user_message(state)

        result = call_structured(system_prompt, user_message, ImpactEvaluation)
        state.impact_result = result
        state.missing_fields_by_agent["impact"] = []
        logger.info("agent_completed", agent="impact", score=result.score)

    except LLMCallError as e:
        logger.error("agent_llm_failed", agent="impact", error=str(e))
        state.missing_fields_by_agent["impact"] = ["llm_error"]

    return state