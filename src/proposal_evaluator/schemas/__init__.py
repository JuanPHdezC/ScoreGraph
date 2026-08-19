from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class Criterio(str, Enum):
    FEASIBILITY = "feasibility"
    IMPACT = "impact"
    COST = "cost"
    NOVELTY = "novelty"


class OperadorRiesgo(str, Enum):
    GT = "gt"
    LT = "lt"
    EQ = "eq"
    IS_TRUE = "is_true"


class AgentEvaluationOutput(BaseModel):
    """Schema base que deben cumplir los 4 agentes evaluadores."""
    score: int = Field(ge=0, le=100, description="Score de 0 a 100")
    summary: str = Field(min_length=1, max_length=500, description="Razonamiento interno breve, NO para mostrar al usuario final")
    confidence_fields_present: bool = Field(description="Indica si el agente considera que tiene campos suficientes para evaluar")


class FeasibilityEvaluation(AgentEvaluationOutput):
    """Evaluación de Factibilidad - campos específicos si aplica."""
    pass


class ImpactEvaluation(AgentEvaluationOutput):
    """Evaluación de Impacto - campos específicos si aplica."""
    pass


class CostEvaluation(AgentEvaluationOutput):
    """Evaluación de Costo - incluye costo estimado."""
    costo_estimado_usd: float | None = Field(default=None, ge=0, description="Costo estimado en USD si se pudo determinar")


class NoveltyEvaluation(AgentEvaluationOutput):
    """Evaluación de Novedad - incluye detección de colisión de patente."""
    colision_patente_detectada: bool = Field(default=False, description="Si se detectó posible colisión con patentes existentes")
    fuentes_consultadas: list[str] = Field(default_factory=list, description="Fuentes consultadas para evaluar novedad")


class Rubric(BaseModel):
    """Schema de una rúbrica configurable por criterio."""
    id: str
    criterio: Criterio
    campos_requeridos: list[str] = Field(default_factory=list, description="Campos obligatorios en la propuesta para evaluar este criterio")
    campos_opcionales: list[str] = Field(default_factory=list, description="Campos opcionales que enriquecen la evaluación")
    criterios_de_evaluacion: str = Field(min_length=1, description="Texto libre que se inyecta al prompt del agente")
    escala_guia: dict[str, str] = Field(default_factory=dict, description="Guía de qué significa cada rango de score, ej: {'0-30': '...', '31-70': '...', '71-100': '...'}")


class RiskRule(BaseModel):
    """Regla de riesgo de negocio determinista para disparar HITL."""
    id: str
    criterio: Criterio
    campo_a_evaluar: str = Field(description="Campo del resultado del agente a evaluar, ej: 'costo_estimado_usd'")
    operador: OperadorRiesgo
    valor_umbral: float | bool | str = Field(description="Valor umbral para comparar")
    descripcion_razon: str = Field(min_length=1, description="Texto para log de auditoría si se dispara")


class CriteriaWeights(BaseModel):
    """Pesos de agregación por criterio (deben sumar 1.0)."""
    feasibility: float = Field(default=0.25, ge=0, le=1)
    impact: float = Field(default=0.25, ge=0, le=1)
    cost: float = Field(default=0.25, ge=0, le=1)
    novelty: float = Field(default=0.25, ge=0, le=1)

    @model_validator(mode="after")
    def validate_sum(self) -> "CriteriaWeights":
        total = self.feasibility + self.impact + self.cost + self.novelty
        if abs(total - 1.0) > 0.001:
            raise ValueError("Los pesos deben sumar 1.0")
        return self


class EvaluationState(BaseModel):
    """Estado completo del grafo de LangGraph."""
    proposal_id: str
    proposal_text: str
    raw_proposal_data: dict[str, Any] = Field(default_factory=dict, description="Campos estructurados si el usuario los proveyó")

    # Resultados por agente (se llenan progresivamente)
    feasibility_result: FeasibilityEvaluation | None = None
    impact_result: ImpactEvaluation | None = None
    cost_result: CostEvaluation | None = None
    novelty_result: NoveltyEvaluation | None = None

    # Campos faltantes detectados por agente
    missing_fields_by_agent: dict[str, list[str]] = Field(default_factory=dict)

    # Control de reintentos y scoring
    retry_count: int = Field(default=0, ge=0)
    weighted_score: float | None = Field(default=None, ge=0, le=100)

    # HITL
    hitl_required: bool = False
    hitl_reason: str | None = None
    # NUEVO: Tipo de HITL para distinguir causa de pausa
    hitl_type: str | None = Field(default=None, description="Tipo de HITL: 'risk' | 'narrative_hallucination' | 'narrative_hallucination_exhausted'")
    # NUEVO: Contador independiente para regeneraciones de narrativa
    narrative_retry_count: int = Field(default=0, ge=0)
    # NUEVO: Decisión de HITL (viene del interrupt)
    hitl_decision: dict | None = Field(default=None, description="Decisión humana del interrupt de HITL")

    # Sintetizador
    final_report: str | None = None
    requiere_revision_manual: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)