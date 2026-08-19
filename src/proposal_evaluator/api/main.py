from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from uuid import uuid4

from proposal_evaluator.schemas import EvaluationState
from proposal_evaluator.graph.build_graph import (
    run_evaluation,
    resume_evaluation,
    get_evaluation_state,
)
from proposal_evaluator.observability.metrics import get_metrics_summary

app = FastAPI(
    title="Proposal Evaluator API",
    description="Sistema Agéntico de Evaluación de Propuestas de Innovación - MVP",
    version="0.1.0",
)


# ─── Request/Response Models ───
class EvaluationRequest(BaseModel):
    proposal_text: str = Field(..., min_length=10, description="Texto libre de la propuesta")
    raw_proposal_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Campos estructurados opcionales")


class EvaluationResponse(BaseModel):
    proposal_id: str
    status: str  # "running", "paused_missing_info", "paused_hitl", "completed", "error"
    message: str
    state: Optional[EvaluationState] = None


class ProvideInfoRequest(BaseModel):
    raw_proposal_data: Dict[str, Any] = Field(..., description="Nueva información para campos faltantes")


class HitlApproveRequest(BaseModel):
    approved: bool = Field(default=True, description="True para aprobar, False para rechazar (para HITL de riesgo)")
    comment: Optional[str] = Field(None, description="Comentario opcional del revisor")
    # NUEVO: Acción específica según hitl_type
    # Para risk: "approve" | "reject" (mapea a approved=true/false)
    # Para narrative_hallucination: "approve_text" | "regenerate"
    action: Optional[str] = Field(
        default=None, 
        description="Acción específica: 'approve'/'reject' (risk) o 'approve_text'/'regenerate' (narrative_hallucination)"
    )


# ─── Endpoints ───
@app.post(
    "/evaluations",
    response_model=EvaluationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Iniciar nueva evaluación de propuesta",
)
async def create_evaluation(request: EvaluationRequest) -> EvaluationResponse:
    """
    Inicia una nueva evaluación de propuesta.
    
    - Genera un proposal_id único
    - Ejecuta el grafo de evaluación (puede pausarse por info faltante o HITL)
    - Devuelve el estado actual y proposal_id para consultas posteriores
    """
    proposal_id = str(uuid4())
    
    try:
        state = run_evaluation(
            proposal_id=proposal_id,
            proposal_text=request.proposal_text,
            raw_proposal_data=request.raw_proposal_data,
        )
        
        # Determinar estado
        if state.requiere_revision_manual:
            status_str = "completed"
            message = "Evaluación completada (requiere revisión manual por alucinación detectada)"
        elif state.hitl_required:
            status_str = "paused_hitl"
            hitl_type = state.hitl_type or "risk"
            if hitl_type == "narrative_hallucination":
                message = f"Evaluación pausada: narrativa con cifras no verificadas - {state.hitl_reason}"
            elif hitl_type == "narrative_hallucination_exhausted":
                message = f"Evaluación pausada: máximo de regeneraciones de narrativa agotado - {state.hitl_reason}"
            else:
                message = f"Evaluación pausada: requiere validación HITL - {state.hitl_reason}"
        elif any(state.missing_fields_by_agent.get(agent) for agent in ["feasibility", "impact", "cost", "novelty"]):
            status_str = "paused_missing_info"
            missing = {k: v for k, v in state.missing_fields_by_agent.items() if v}
            message = f"Evaluación pausada: faltan campos - {missing}"
        else:
            status_str = "completed"
            message = "Evaluación completada exitosamente"
        
        return EvaluationResponse(
            proposal_id=proposal_id,
            status=status_str,
            message=message,
            state=state,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error iniciando evaluación: {str(e)}"
        )


@app.get(
    "/evaluations/{proposal_id}",
    response_model=EvaluationResponse,
    summary="Obtener estado de una evaluación",
)
async def get_evaluation(proposal_id: str) -> EvaluationResponse:
    """
    Obtiene el estado actual de una evaluación.
    
    - Si está completada, incluye el reporte final
    - Si está pausada, indica el motivo (info faltante o HITL)
    """
    state = get_evaluation_state(proposal_id)
    
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluación {proposal_id} no encontrada"
        )
    
    # Determinar estado
    if state.requiere_revision_manual:
        status_str = "completed"
        message = "Evaluación completada (requiere revisión manual)"
    elif state.hitl_required:
        status_str = "paused_hitl"
        hitl_type = state.hitl_type or "risk"
        if hitl_type == "narrative_hallucination":
            message = f"Pausada por narrativa con cifras no verificadas: {state.hitl_reason}"
        elif hitl_type == "narrative_hallucination_exhausted":
            message = f"Pausada: máximo de regeneraciones de narrativa agotado - {state.hitl_reason}"
        else:
            message = f"Pausada por HITL de riesgo: {state.hitl_reason}"
    elif any(state.missing_fields_by_agent.get(agent) for agent in ["feasibility", "impact", "cost", "novelty"]):
        status_str = "paused_missing_info"
        missing = {k: v for k, v in state.missing_fields_by_agent.items() if v}
        message = f"Pausada por campos faltantes: {missing}"
    else:
        status_str = "completed"
        message = "Evaluación completada"
    
    return EvaluationResponse(
        proposal_id=proposal_id,
        status=status_str,
        message=message,
        state=state,
    )


@app.post(
    "/evaluations/{proposal_id}/provide-info",
    response_model=EvaluationResponse,
    summary="Proveer información faltante para reanudar evaluación",
)
async def provide_info(proposal_id: str, request: ProvideInfoRequest) -> EvaluationResponse:
    """
    Reanuda una evaluación pausada por información faltante.
    
    - Envía los campos faltantes en raw_proposal_data
    - Reanuda el grafo desde el punto de pausa
    """
    # Verificar que existe
    existing = get_evaluation_state(proposal_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluación {proposal_id} no encontrada"
        )
    
    if not existing.hitl_required and not any(existing.missing_fields_by_agent.get(a) for a in ["feasibility", "impact", "cost", "novelty"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La evaluación no está pausada esperando información"
        )
    
    try:
        state = resume_evaluation(
            proposal_id=proposal_id,
            new_data={"raw_proposal_data": request.raw_proposal_data},
        )
        
        # Determinar nuevo estado
        if state.requiere_revision_manual:
            status_str = "completed"
            message = "Evaluación completada (requiere revisión manual)"
        elif state.hitl_required:
            status_str = "paused_hitl"
            hitl_type = state.hitl_type or "risk"
            if hitl_type == "narrative_hallucination":
                message = f"Evaluación pausada: narrativa con cifras no verificadas - {state.hitl_reason}"
            elif hitl_type == "narrative_hallucination_exhausted":
                message = f"Evaluación pausada: máximo de regeneraciones de narrativa agotado - {state.hitl_reason}"
            else:
                message = f"Evaluación pausada: requiere validación HITL - {state.hitl_reason}"
        elif any(state.missing_fields_by_agent.get(agent) for agent in ["feasibility", "impact", "cost", "novelty"]):
            status_str = "paused_missing_info"
            missing = {k: v for k, v in state.missing_fields_by_agent.items() if v}
            message = f"Evaluación pausada: aún faltan campos - {missing}"
        else:
            status_str = "completed"
            message = "Evaluación completada exitosamente"
        
        return EvaluationResponse(
            proposal_id=proposal_id,
            status=status_str,
            message=message,
            state=state,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reanudando evaluación: {str(e)}"
        )


@app.post(
    "/evaluations/{proposal_id}/hitl-approve",
    response_model=EvaluationResponse,
    summary="Aprobar/rechazar evaluación pausada por HITL",
)
async def hitl_approve(proposal_id: str, request: HitlApproveRequest) -> EvaluationResponse:
    """
    Permite a un humano resolver una evaluación pausada por HITL.
    
    Acciones según hitl_type:
    - risk (riesgo de negocio): action="approve" | "reject" (mapea a approved=true/false)
    - narrative_hallucination: action="approve_text" | "regenerate"
    - narrative_hallucination_exhausted: solo lectura, no permite acción
    
    El campo action es obligatorio para narrative_hallucination.
    Para risk, si no se proporciona action, usa approved (compatibilidad hacia atrás).
    """
    existing = get_evaluation_state(proposal_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluación {proposal_id} no encontrada"
        )
    
    if not existing.hitl_required:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La evaluación no está pausada por HITL"
        )
    
    hitl_type = existing.hitl_type or "risk"  # Compatibilidad: legacy states default to "risk"
    
    # Validar acción según hitl_type
    action = request.action
    
    if hitl_type == "narrative_hallucination":
        valid_actions = ["approve_text", "regenerate"]
        if action not in valid_actions:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": f"Acción '{action}' no válida para hitl_type='narrative_hallucination'",
                    "valid_actions": valid_actions,
                    "hitl_type": hitl_type,
                }
            )
        # Mapear action a decision_data para el grafo
        decision_data = {
            "action": action,
            "comment": request.comment or "",
        }
        
    elif hitl_type in ("risk", None):
        # Compatibilidad: si no hay action, usar approved (legacy)
        if action is None:
            action = "approve" if request.approved else "reject"
        valid_actions = ["approve", "reject"]
        if action not in valid_actions:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": f"Acción '{action}' no válida para hitl_type='risk'",
                    "valid_actions": valid_actions,
                    "hitl_type": "risk",
                }
            )
        decision_data = {
            "approved": action == "approve",
            "comment": request.comment or "",
        }
        
    elif hitl_type == "narrative_hallucination_exhausted":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La evaluación ha agotado los reintentos de regeneración de narrativa. No se puede resolver vía API."
        )
        
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tipo de HITL desconocido: {hitl_type}"
        )
    
    try:
        state = resume_evaluation(
            proposal_id=proposal_id,
            new_data=decision_data,
        )
        
        # Determinar estado de respuesta
        if state.requiere_revision_manual:
            status_str = "completed"
            message = "Evaluación completada (requiere revisión manual)"
        elif state.hitl_required:
            status_str = "paused_hitl"
            message = f"Evaluación pausada: {state.hitl_reason}"
        elif any(state.missing_fields_by_agent.get(agent) for agent in ["feasibility", "impact", "cost", "novelty"]):
            status_str = "paused_missing_info"
            missing = {k: v for k, v in state.missing_fields_by_agent.items() if v}
            message = f"Evaluación pausada: aún faltan campos - {missing}"
        else:
            status_str = "completed"
            message = "Evaluación completada exitosamente"
        
        return EvaluationResponse(
            proposal_id=proposal_id,
            status=status_str,
            message=message,
            state=state,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando decisión HITL: {str(e)}"
        )


@app.get("/health", summary="Health check")
async def health_check():
    return {"status": "ok", "service": "proposal-evaluator"}


@app.get("/metrics/summary", summary="Métricas agregadas de evaluaciones")
async def metrics_summary():
    """
    Devuelve métricas agregadas sobre todas las evaluaciones.
    
    Incluye:
    - Total evaluaciones completadas
    - Tasa de HITL disparado
    - Tasa de reintentos por info faltante
    - Scores promedio por criterio
    """
    return get_metrics_summary()