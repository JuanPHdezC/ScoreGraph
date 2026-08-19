import structlog
from typing import Literal
from langgraph.types import interrupt
from proposal_evaluator.config.config_repository import log_audit_event
from proposal_evaluator.schemas import EvaluationState

logger = structlog.get_logger(__name__)

MAX_RETRIES = 3


def build_missing_info_payload(state: EvaluationState) -> dict:
    """
    Construye el payload de solicitud de información faltante para el usuario.
    """
    missing_by_agent = state.missing_fields_by_agent
    payload = {
        "proposal_id": state.proposal_id,
        "message": "Se requiere información adicional para completar la evaluación.",
        "missing_fields_by_agent": {
            agent: fields for agent, fields in missing_by_agent.items() if fields
        },
        "retry_count": state.retry_count,
        "max_retries": MAX_RETRIES,
    }
    return payload


def missing_info_node(state: EvaluationState) -> EvaluationState:
    """
    Nodo de solicitud de información faltante.
    - Si hay campos faltantes y retry_count < MAX_RETRIES: pausa con interrupt()
    - Si retry_count >= MAX_RETRIES: fuerza hitl_required=True
    - Loggea la solicitud como evento de auditoría.
    """
    proposal_id = state.proposal_id

    # Filtrar agentes con campos faltantes
    agents_with_missing = {
        agent: fields for agent, fields in state.missing_fields_by_agent.items() if fields
    }

    if not agents_with_missing:
        # No hay campos faltantes, continuar normal
        logger.info("missing_info_no_action", proposal_id=proposal_id)
        return state

    # Incrementar contador de reintentos
    state.retry_count += 1

    # Log de auditoría
    log_audit_event(
        proposal_id=proposal_id,
        event_type="missing_info_requested",
        event_data={
            "missing_fields_by_agent": agents_with_missing,
            "retry_count": state.retry_count,
            "max_retries": MAX_RETRIES,
        },
    )

    logger.warning(
        "missing_info_request",
        proposal_id=proposal_id,
        missing_fields=agents_with_missing,
        retry_count=state.retry_count,
    )

    if state.retry_count >= MAX_RETRIES:
        # Forzar HITL tras agotar reintentos
        state.hitl_required = True
        state.hitl_reason = "info_incompleta_tras_reintentos"
        log_audit_event(
            proposal_id=proposal_id,
            event_type="hitl_triggered",
            event_data={
                "reason": "max_retries_exceeded",
                "retry_count": state.retry_count,
                "missing_fields_by_agent": agents_with_missing,
            },
        )
        logger.error(
            "missing_info_max_retries_exceeded",
            proposal_id=proposal_id,
            retry_count=state.retry_count,
        )
        return state

    # Preparar payload para el usuario
    payload = build_missing_info_payload(state)

    # PAUSAR el grafo con interrupt - espera respuesta humana
    # El valor devuelto por interrupt() será lo que pase resume_evaluation()
    user_response = interrupt(payload)

    # Cuando se reanuda, user_response contiene los datos que el usuario proveyó
    # Actualizamos raw_proposal_data con la nueva información
    if user_response and isinstance(user_response, dict):
        new_data = user_response.get("raw_proposal_data", {})
        if new_data:
            state.raw_proposal_data.update(new_data)
            logger.info(
                "missing_info_resumed",
                proposal_id=proposal_id,
                new_fields=list(new_data.keys()),
            )

    # Limpiar missing_fields para que los agentes re-evalúen
    state.missing_fields_by_agent = {}

    return state


def should_request_missing_info(state: EvaluationState) -> Literal["request_info", "continue"]:
    """
    Función de decisión condicional para el grafo.
    Retorna 'request_info' si hay campos faltantes, 'continue' si no.
    """
    has_missing = any(
        fields for fields in state.missing_fields_by_agent.values()
    )
    return "request_info" if has_missing else "continue"