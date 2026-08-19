import re
import structlog
from typing import Optional
from proposal_evaluator.llm.client import call_structured, LLMCallError
from proposal_evaluator.schemas import EvaluationState
from proposal_evaluator.config.config_repository import log_audit_event
from langgraph.types import interrupt

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """Eres un analista senior de innovación redactando un reporte narrativo para comité ejecutivo.

Tu ÚNICA tarea: redactar una narrativa cualitativa explicando fortalezas, debilidades, riesgos y oportunidades de la propuesta de innovación.

REGLAS ESTRICTAS (INCUMPLIBLES):
1. NO incluyas NINGÚN número, cifra, porcentaje, score, moneda, o valor numérico en tu respuesta.
2. NO uses placeholders como [score], {costo}, $X, N%, etc.
3. NO calcules, infieras, compares ni predigas valores numéricos.
4. Tu salida es SOLO texto narrativo en español, prosa profesional, sin listas numeradas con cifras.
5. Los números (scores, costos, etc.) se insertarán AUTOMÁTICAMENTE por el sistema en el template final.

ESTRUCTURA SUGERIDA DEL TEXTO:
- Párrafo de apertura: resumen ejecutivo cualitativo de la propuesta
- Fortalezas: qué aspectos son sólidos y por qué (cualitativo)
- Debilidades/riesgos: qué aspectos requieren atención y por qué (cualitativo)
- Conclusión: recomendación general (aprobar, condicionar, rechazar) con justificación cualitativa

Recibirás los scores y cifras en el contexto, pero DEBES ignorarlos para tu redacción y centrarte en el análisis cualitativo."""

MAX_NARRATIVE_RETRIES = 3


def extract_numbers(text: str) -> list[str]:
    """Extrae todos los números (enteros, decimales, porcentajes, moneda) del texto."""
    pattern = r'(?:\$|USD\s)?\d+(?:[.,]\d+)?%?(?:\s*[-–]\s*(?:\$|USD\s)?\d+(?:[.,]\d+)?%?)?'
    return re.findall(pattern, text)


def validate_numbers_in_narrative(
    narrative: str,
    allowed_values: dict,
    proposal_id: str
) -> tuple[bool, list[str]]:
    """
    Valida que todos los números en el texto narrativo coincidan con valores permitidos.
    
    Returns:
        (is_valid, unrecognized_numbers)
    """
    found_numbers = extract_numbers(narrative)
    
    allowed_set = set()
    for key, value in allowed_values.items():
        if value is not None:
            allowed_set.add(str(value))
            if isinstance(value, float):
                allowed_set.add(f"{value:.2f}")
                allowed_set.add(f"{value:.1f}")
                allowed_set.add(f"{int(value)}")
            if isinstance(value, (int, float)):
                allowed_set.add(f"{value}%")
                allowed_set.add(f"${value}")
                allowed_set.add(f"USD {value}")
    
    unrecognized = []
    for num in found_numbers:
        normalized = num.replace(',', '').replace('$', '').replace('USD', '').replace('%', '').strip()
        matched = False
        for allowed in allowed_set:
            allowed_norm = allowed.replace(',', '').replace('$', '').replace('USD', '').replace('%', '').strip()
            if normalized == allowed_norm:
                matched = True
                break
        if not matched:
            unrecognized.append(num)
    
    return len(unrecognized) == 0, unrecognized


def build_synthesizer_prompt(state: EvaluationState, correction_context: str = "") -> str:
    """Construye el prompt para el sintetizador con contexto de la evaluación."""
    parts = [
        "CONTEXTO DE LA EVALUACIÓN (SOLO PARA TU ANÁLISIS CUALITATIVO, NO PARA CITAR CIFRAS):",
        f"Propuesta: {state.proposal_text[:500]}...",
        "",
        "SCORES DE LOS 4 CRITERIOS (referencia cualitativa):",
        f"- Factibilidad: {state.feasibility_result.score if state.feasibility_result else 'N/A'}/100 - {state.feasibility_result.summary if state.feasibility_result else 'N/A'}",
        f"- Impacto: {state.impact_result.score if state.impact_result else 'N/A'}/100 - {state.impact_result.summary if state.impact_result else 'N/A'}",
        f"- Costo: {state.cost_result.score if state.cost_result else 'N/A'}/100 - {state.cost_result.summary if state.cost_result else 'N/A'}",
        f"- Novedad: {state.novelty_result.score if state.novelty_result else 'N/A'}/100 - {state.novelty_result.summary if state.novelty_result else 'N/A'}",
        "",
        f"COSTO ESTIMADO: ${state.cost_result.costo_estimado_usd:,.0f}" if state.cost_result and state.cost_result.costo_estimado_usd else "COSTO ESTIMADO: No determinado",
        f"COLISIÓN DE PATENTE: {'Sí detectada' if state.novelty_result and state.novelty_result.colision_patente_detectada else 'No detectada'}",
        f"FUENTES CONSULTADA (Novedad): {', '.join(state.novelty_result.fuentes_consultadas) if state.novelty_result and state.novelty_result.fuentes_consultadas else 'Ninguna'}",
        "",
        f"SCORE PONDERADO FINAL: {state.weighted_score}/100" if state.weighted_score else "SCORE PONDERADO: Pendiente",
        f"HITL REQUERIDO: {'Sí - ' + state.hitl_reason if state.hitl_required else 'No'}",
        "",
    ]
    
    if correction_context:
        parts.append(correction_context)
        parts.append("")
    
    parts.append("INSTRUCCIÓN: Redacta ÚNICAMENTE la narrativa cualitativa. NO incluyas números.")
    return "\n".join(parts)


def _log_hallucination_detected(proposal_id: str, unrecognized: list[str], qualitative_text: str, allowed_values: dict):
    """Loggea evento de auditoría con contexto del número no reconocido."""
    # Encontrar contexto alrededor del número no reconocido
    context_snippets = []
    for num in unrecognized:
        idx = qualitative_text.find(num)
        if idx >= 0:
            start = max(0, idx - 80)
            end = min(len(qualitative_text), idx + len(num) + 80)
            context_snippets.append({
                "unrecognized_number": num,
                "surrounding_text": qualitative_text[start:end],
                "position": idx,
            })
    
    log_audit_event(
        proposal_id=proposal_id,
        event_type="narrative_hallucination_detected",
        event_data={
            "unrecognized_numbers": unrecognized,
            "context_snippets": context_snippets,
            "full_narrative_preview": qualitative_text[:300],
            "allowed_values": {k: v for k, v in allowed_values.items() if v is not None},
        },
    )
    logger.warning(
        "synthesizer_hallucination_detected",
        proposal_id=proposal_id,
        unrecognized=unrecognized,
    )


def synthesizer_agent(state: EvaluationState) -> EvaluationState:
    """
    Nodo Sintetizador con soporte para pausa por alucinación y regeneración.
    
    Flujo:
    1. Si venimos de un interrupt (state contiene 'hitl_decision'), procesar la decisión humana
    2. Si es primera ejecución o regeneración, llamar LLM y validar
    3. Si hay alucinación → pausar con interrupt() 
    4. Si no hay alucinación → ensamblar reporte y continuar
    """
    proposal_id = state.proposal_id
    logger.info("synthesizer_start", proposal_id=proposal_id, 
                hitl_type=state.hitl_type, 
                narrative_retry_count=state.narrative_retry_count)

    # Verificar si venimos de un interrupt de narrativa HITL
    hitl_decision = getattr(state, 'hitl_decision', None)
    
    if hitl_decision and state.hitl_type == "narrative_hallucination":
        # Procesar decisión humana tras pausa por alucinación
        action = hitl_decision.get("action")
        logger.info("synthesizer_resuming_from_hitl", proposal_id=proposal_id, action=action)
        
        if action == "approve_text":
            # Humano aprobó el texto tal cual → ensamblar reporte con narrativa original
            qualitative_text = state.hitl_decision.get("qualitative_text", "")
            report = build_final_report(state, qualitative_text)
            state.final_report = report
            state.hitl_required = False
            state.hitl_type = None
            state.hitl_reason = None
            state.requiere_revision_manual = True  # Marcar que requirió revisión humana
            log_audit_event(
                proposal_id=proposal_id,
                event_type="narrative_hitl_approved",
                event_data={"action": "approve_text", "retry_count": state.narrative_retry_count},
            )
            logger.info("narrative_hitl_approved", proposal_id=proposal_id)
            return state
            
        elif action == "regenerate":
            # Humano pidió regeneración
            if state.narrative_retry_count >= 3:
                # Límite agotado
                state.hitl_type = "narrative_hallucination_exhausted"
                state.hitl_required = True
                state.hitl_reason = "Máximo de regeneraciones de narrativa agotado (3)"
                state.final_report = None
                log_audit_event(
                    proposal_id=proposal_id,
                    event_type="narrative_hallucination_exhausted",
                    event_data={"retry_count": state.narrative_retry_count},
                )
                logger.error("narrative_hallucination_exhausted", proposal_id=proposal_id)
                return state
            
            # Incrementar contador y regenerar
            state.narrative_retry_count += 1
            state.hitl_required = False
            state.hitl_type = None
            state.hitl_reason = None
            state.final_report = None
            # Continuar a regeneración normal (caer al flujo normal abajo)
            logger.info("narrative_regeneration_requested", proposal_id=proposal_id, 
                       retry_count=state.narrative_retry_count)
            # Continuar flujo normal para regeneración
    
    # Verificar estado previo completo
    if not all([state.feasibility_result, state.impact_result, state.cost_result, state.novelty_result]):
        logger.warning("synthesizer_incomplete_state", proposal_id=proposal_id)
        state.final_report = "Error: Estado incompleto para generar reporte"
        state.requiere_revision_manual = True
        return state

    try:
        # Construir prompt con contexto (incluye corrección si viene de regeneración)
        correction_context = ""
        if state.narrative_retry_count > 0:
            unrecognized = state.hitl_decision.get("unrecognized_numbers", []) if state.hitl_decision else []
            if unrecognized:
                correction_context = (
                    f"CORRECCIÓN REQUERIDA: En la narrativa anterior se detectó el número no permitido: {', '.join(unrecognized)}. "
                    f"Este número NO está permitido. Reescribe la narrativa SIN incluir ese número ni cifras similares. "
                    f"Recuerda: SOLO texto cualitativo, NINGÚN número."
                )
        
        user_prompt = build_synthesizer_prompt(state, correction_context)
        
        # Llamar LLM para narrativa cualitativa
        from pydantic import BaseModel
        
        class NarrativeOutput(BaseModel):
            narrative: str
        
        result = call_structured(SYSTEM_PROMPT, user_prompt, NarrativeOutput)
        qualitative_text = result.narrative.strip()

        # Validación anti-alucinación
        allowed_values = {
            "feasibility_score": state.feasibility_result.score,
            "impact_score": state.impact_result.score,
            "cost_score": state.cost_result.score,
            "novelty_score": state.novelty_result.score,
            "weighted_score": state.weighted_score,
            "costo_estimado_usd": state.cost_result.costo_estimado_usd,
        }
        
        is_valid, unrecognized = validate_numbers_in_narrative(
            qualitative_text, allowed_values, proposal_id
        )

        if not is_valid:
            # Log de auditoría con contexto
            _log_hallucination_detected(proposal_id, unrecognized, qualitative_text, 
                                       {k: v for k, v in allowed_values.items() if v is not None})
            
            # PAUSAR el grafo con interrupt para HITL por alucinación
            state.hitl_required = True
            state.hitl_type = "narrative_hallucination"
            state.hitl_reason = f"Narrativa contiene cifras no verificadas: {', '.join(unrecognized)}"
            state.final_report = None  # NO ensamblar reporte
            state.requiere_revision_manual = True
            
            # Guardar narrativa para posible aprobación posterior
            payload = {
                "proposal_id": proposal_id,
                "message": "La narrativa generada contiene cifras no verificadas. Revisión humana requerida.",
                "hitl_type": "narrative_hallucination",
                "hitl_reason": state.hitl_reason,
                "unrecognized_numbers": unrecognized,
                "qualitative_text": qualitative_text,  # Guardar para posible aprobación
                "allowed_values": {k: v for k, v in allowed_values.items() if v is not None},
                "narrative_preview": qualitative_text[:300],
            }
            
            log_audit_event(
                proposal_id=proposal_id,
                event_type="narrative_hitl_paused",
                event_data=payload,
            )
            logger.warning("narrative_hitl_paused", proposal_id=proposal_id, unrecognized=unrecognized)
            
            # PAUSAR con interrupt - espera decisión humana
            hitl_decision = interrupt(payload)
            
            # Al reanudar, el estado vendrá con hitl_decision
            state.hitl_decision = hitl_decision
            return state

        # Sin alucinación → ensamblar reporte final
        report = build_final_report(state, qualitative_text)
        state.final_report = report
        state.requiere_revision_manual = False
        state.hitl_required = False
        state.hitl_type = None
        state.hitl_reason = None
        
        log_audit_event(
            proposal_id=proposal_id,
            event_type="synthesizer_completed",
            event_data={
                "requires_manual_review": False,
                "narrative_length": len(qualitative_text),
            },
        )
        logger.info("synthesizer_completed", proposal_id=proposal_id)

    except LLMCallError as e:
        logger.error("synthesizer_llm_failed", proposal_id=proposal_id, error=str(e))
        state.final_report = f"Error generando reporte: {e}"
        state.requiere_revision_manual = True
    except Exception as e:
        logger.error("synthesizer_error", proposal_id=proposal_id, error=str(e))
        state.final_report = f"Error inesperado en sintetizador: {e}"
        state.requiere_revision_manual = True

    return state


def _log_hallucination_detected(proposal_id: str, unrecognized: list[str], qualitative_text: str, allowed_values: dict):
    """Loggea evento de auditoría con contexto del número no reconocido."""
    context_snippets = []
    for num in unrecognized:
        idx = qualitative_text.find(num)
        if idx >= 0:
            start = max(0, idx - 80)
            end = min(len(qualitative_text), idx + len(num) + 80)
            context_snippets.append({
                "unrecognized_number": num,
                "surrounding_text": qualitative_text[start:end],
                "position": idx,
            })
    
    log_audit_event(
        proposal_id=proposal_id,
        event_type="narrative_hallucination_detected",
        event_data={
            "unrecognized_numbers": unrecognized,
            "context_snippets": context_snippets,
            "full_narrative_preview": qualitative_text[:300],
            "allowed_values": {k: v for k, v in allowed_values.items() if v is not None},
        },
    )
    logger.warning(
        "synthesizer_hallucination_detected",
        proposal_id=proposal_id,
        unrecognized=unrecognized,
    )


def build_final_report(state: EvaluationState, qualitative_text: str) -> str:
    """Ensambla el reporte final combinando template con números exactos + narrativa cualitativa."""
    lines = [
        "=" * 60,
        "REPORTE DE EVALUACIÓN DE PROPUESTA DE INNOVACIÓN",
        "=" * 60,
        "",
        f"ID Propuesta: {state.proposal_id}",
        f"Texto original: {state.proposal_text[:200]}..." if len(state.proposal_text) > 200 else f"Texto original: {state.proposal_text}",
        "",
        "─" * 60,
        "SCORES DE EVALUACIÓN",
        "─" * 60,
        f"  Factibilidad:     {state.feasibility_result.score:3d}/100" if state.feasibility_result else "  Factibilidad:     N/A",
        f"  Impacto:          {state.impact_result.score:3d}/100" if state.impact_result else "  Impacto:          N/A",
        f"  Costo:            {state.cost_result.score:3d}/100" if state.cost_result else "  Costo:            N/A",
        f"  Novedad:          {state.novelty_result.score:3d}/100" if state.novelty_result else "  Novedad:          N/A",
        "",
        f"  SCORE PONDERADO:  {state.weighted_score:.2f}/100" if state.weighted_score else "  SCORE PONDERADO:  N/A",
        "",
        "─" * 60,
        "DETALLES ADICIONALES",
        "─" * 60,
    ]
    
    if state.cost_result and state.cost_result.costo_estimado_usd:
        lines.append(f"  Costo estimado:   ${state.cost_result.costo_estimado_usd:,.2f} USD")
    else:
        lines.append("  Costo estimado:   No determinado")
    
    if state.novelty_result:
        lines.append(f"  Colisión patente: {'SÍ' if state.novelty_result.colision_patente_detectada else 'NO'}")
        if state.novelty_result.fuentes_consultadas:
            lines.append(f"  Fuentes consultadas: {', '.join(state.novelty_result.fuentes_consultadas)}")
    
    if state.hitl_required:
        lines.append(f"  HITL REQUERIDO:   SÍ - {state.hitl_reason}")
    else:
        lines.append("  HITL REQUERIDO:   No")
    
    lines.extend([
        "",
        "─" * 60,
        "ANÁLISIS CUALITATIVO (generado por LLM)",
        "─" * 60,
        "",
        qualitative_text,
        "",
        "=" * 60,
        "FIN DEL REPORTE",
        "=" * 60,
    ])
    
    if state.requiere_revision_manual:
        lines.insert(-2, "\n⚠️  ADVERTENCIA: Este reporte requirió revisión manual.")
        lines.insert(-2, "     Se detectaron cifras en la narrativa no verificadas contra la evaluación.")
    
    return "\n".join(lines)