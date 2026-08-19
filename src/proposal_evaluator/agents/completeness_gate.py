import structlog
from proposal_evaluator.schemas import Rubric

logger = structlog.get_logger(__name__)


def check_completeness(
    proposal_id: str,
    criterio: str,
    required_fields: list[str],
    raw_proposal_data: dict,
) -> list[str]:
    """
    Gate determinista de completitud: verifica qué campos requeridos faltan en raw_proposal_data.
    NO invoca ningún LLM. Es la ÚNICA fuente de verdad sobre suficiencia de información.

    Args:
        proposal_id: ID de la propuesta para logging
        criterio: Nombre del criterio (feasibility, impact, cost, novelty)
        required_fields: Lista de campos requeridos desde la Rubric
        raw_proposal_data: Datos estructurados provistos por el usuario

    Returns:
        Lista de campos requeridos que están ausentes (vacía si todo presente)
    """
    missing = []
    for field in required_fields:
        if field not in raw_proposal_data or raw_proposal_data[field] in (None, ""):
            missing.append(field)

    if missing:
        logger.warning(
            "completeness_gate_missing_fields",
            proposal_id=proposal_id,
            criterio=criterio,
            missing_fields=missing,
            required_fields=required_fields,
        )
    else:
        logger.info(
            "completeness_gate_ok",
            proposal_id=proposal_id,
            criterio=criterio,
            required_fields=required_fields,
        )

    return missing


def check_completeness_from_rubric(
    proposal_id: str,
    rubric: Rubric,
    raw_proposal_data: dict,
) -> list[str]:
    """
    Wrapper que usa directamente una Rubric para el gate de completitud.
    """
    return check_completeness(
        proposal_id=proposal_id,
        criterio=rubric.criterio.value,
        required_fields=rubric.campos_requeridos,
        raw_proposal_data=raw_proposal_data,
    )