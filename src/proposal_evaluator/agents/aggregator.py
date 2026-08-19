import structlog
from typing import Optional
from proposal_evaluator.config.config_repository import (
    get_criteria_weights,
    log_audit_event,
)
from proposal_evaluator.schemas import (
    EvaluationState,
    Criterio,
)

logger = structlog.get_logger(__name__)


def calculate_weighted_score(state: EvaluationState) -> EvaluationState:
    """
    Agregador determinista: calcula weighted_score a partir de los 4 resultados + pesos.
    Sin LLM. Loggea el cálculo como evento de auditoría.
    """
    proposal_id = state.proposal_id
    weights = get_criteria_weights()

    # Verificar que todos los resultados están presentes
    results = {
        Criterio.FEASIBILITY: state.feasibility_result,
        Criterio.IMPACT: state.impact_result,
        Criterio.COST: state.cost_result,
        Criterio.NOVELTY: state.novelty_result,
    }

    missing = [c.value for c, r in results.items() if r is None]
    if missing:
        logger.warning(
            "aggregator_missing_results",
            proposal_id=proposal_id,
            missing_criteria=missing,
        )
        state.weighted_score = None
        return state

    # Calcular score ponderado
    score_feasibility = results[Criterio.FEASIBILITY].score
    score_impact = results[Criterio.IMPACT].score
    score_cost = results[Criterio.COST].score
    score_novelty = results[Criterio.NOVELTY].score

    weighted = (
        score_feasibility * weights.feasibility
        + score_impact * weights.impact
        + score_cost * weights.cost
        + score_novelty * weights.novelty
    )

    state.weighted_score = round(weighted, 2)

    # Log de auditoría estructurado
    log_audit_event(
        proposal_id=proposal_id,
        event_type="score_calculated",
        event_data={
            "individual_scores": {
                "feasibility": score_feasibility,
                "impact": score_impact,
                "cost": score_cost,
                "novelty": score_novelty,
            },
            "weights_used": {
                "feasibility": weights.feasibility,
                "impact": weights.impact,
                "cost": weights.cost,
                "novelty": weights.novelty,
            },
            "weighted_score": state.weighted_score,
        },
    )

    logger.info(
        "aggregator_completed",
        proposal_id=proposal_id,
        weighted_score=state.weighted_score,
    )

    return state