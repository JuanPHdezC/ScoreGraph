import structlog
import json
from typing import Any
from proposal_evaluator.config.config_repository import (
    get_active_risk_rules,
    log_audit_event,
)
from proposal_evaluator.schemas import (
    EvaluationState,
    RiskRule,
    OperadorRiesgo,
)

logger = structlog.get_logger(__name__)


def _get_field_value(state: EvaluationState, campo: str) -> Any:
    """
    Extrae el valor del campo indicado desde el estado consolidado.
    Soporta: score de cada agente, costo_estimado_usd, colision_patente_detectada.
    """
    # Mapeo campo -> extractor
    if campo == "score":
        # Para reglas que aplican sobre 'score' genérico, evaluar el score del agente del criterio
        # La regla ya tiene criterio, así que usamos el agente correspondiente
        pass  # Se maneja en evaluate_risk_rules

    if campo == "costo_estimado_usd":
        return state.cost_result.costo_estimado_usd if state.cost_result else None

    if campo == "colision_patente_detectada":
        return state.novelty_result.colision_patente_detectada if state.novelty_result else False

    if campo in ("feasibility_score", "impact_score", "cost_score", "novelty_score"):
        criterio_map = {
            "feasibility_score": state.feasibility_result,
            "impact_score": state.impact_result,
            "cost_score": state.cost_result,
            "novelty_score": state.novelty_result,
        }
        result = criterio_map.get(campo)
        return result.score if result else None

    return None


def _evaluate_condition(
    field_value: Any,
    operator: OperadorRiesgo,
    threshold: Any,
) -> bool:
    """
    Evalúa una condición determinista: field_value OP threshold.
    """
    if field_value is None:
        return False

    try:
        if operator == OperadorRiesgo.GT:
            return float(field_value) > float(threshold)
        elif operator == OperadorRiesgo.LT:
            return float(field_value) < float(threshold)
        elif operator == OperadorRiesgo.EQ:
            return field_value == threshold
        elif operator == OperadorRiesgo.IS_TRUE:
            return bool(field_value) is True
    except (ValueError, TypeError):
        return False

    return False


def evaluate_risk_rules(state: EvaluationState) -> EvaluationState:
    """
    Evaluador de reglas de riesgo: revisa todas las RiskRule activas contra el estado.
    Si alguna se dispara, marca hitl_required=True y hitl_reason con la descripción.
    Loggea cada regla evaluada y las que se disparan.
    """
    proposal_id = state.proposal_id
    rules = get_active_risk_rules()

    triggered_rules = []

    for rule in rules:
        # Obtener valor del campo según la regla
        if rule.campo_a_evaluar == "score":
            # Score del agente correspondiente al criterio de la regla
            agent_result_map = {
                "feasibility": state.feasibility_result,
                "impact": state.impact_result,
                "cost": state.cost_result,
                "novelty": state.novelty_result,
            }
            agent_result = agent_result_map.get(rule.criterio.value)
            field_value = agent_result.score if agent_result else None
        else:
            field_value = _get_field_value(state, rule.campo_a_evaluar)

        triggered = _evaluate_condition(field_value, rule.operador, rule.valor_umbral)

        logger.info(
            "risk_rule_evaluated",
            proposal_id=proposal_id,
            rule_id=rule.id,
            criterio=rule.criterio.value,
            campo=rule.campo_a_evaluar,
            operador=rule.operador.value,
            field_value=field_value,
            threshold=rule.valor_umbral,
            triggered=triggered,
        )

        if triggered:
            triggered_rules.append(rule)

    if triggered_rules:
        # Tomar la primera regla disparada (o concatenar razones)
        reasons = [r.descripcion_razon for r in triggered_rules]
        state.hitl_required = True
        state.hitl_reason = "; ".join(reasons)

        log_audit_event(
            proposal_id=proposal_id,
            event_type="hitl_triggered",
            event_data={
                "triggered_rules": [
                    {
                        "rule_id": r.id,
                        "criterio": r.criterio.value,
                        "campo": r.campo_a_evaluar,
                        "operador": r.operador.value,
                        "threshold": r.valor_umbral,
                        "reason": r.descripcion_razon,
                    }
                    for r in triggered_rules
                ],
                "hitl_reason": state.hitl_reason,
            },
        )

        logger.warning(
            "risk_gate_hitl_required",
            proposal_id=proposal_id,
            hitl_reason=state.hitl_reason,
            triggered_count=len(triggered_rules),
        )
    else:
        state.hitl_required = False
        state.hitl_reason = None
        logger.info("risk_gate_ok", proposal_id=proposal_id)

    return state