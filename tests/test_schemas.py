import pytest
from proposal_evaluator.schemas import (
    AgentEvaluationOutput,
    FeasibilityEvaluation,
    ImpactEvaluation,
    CostEvaluation,
    NoveltyEvaluation,
    Rubric,
    RiskRule,
    CriteriaWeights,
    EvaluationState,
    Criterio,
    OperadorRiesgo,
)
from pydantic import ValidationError


class TestAgentEvaluationOutput:
    """Tests para el schema base de evaluación de agentes."""

    def test_valid_score(self):
        """Score válido entre 0 y 100 debe pasar."""
        output = AgentEvaluationOutput(score=85, summary="Buena evaluación", confidence_fields_present=True)
        assert output.score == 85

    def test_score_too_low(self):
        """Score < 0 debe fallar."""
        with pytest.raises(ValidationError) as exc_info:
            AgentEvaluationOutput(score=-1, summary="Test", confidence_fields_present=True)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_score_too_high(self):
        """Score > 100 debe fallar."""
        with pytest.raises(ValidationError) as exc_info:
            AgentEvaluationOutput(score=101, summary="Test", confidence_fields_present=True)
        assert "less than or equal to 100" in str(exc_info.value)

    def test_empty_summary_fails(self):
        """Summary vacío debe fallar."""
        with pytest.raises(ValidationError) as exc_info:
            AgentEvaluationOutput(score=50, summary="", confidence_fields_present=True)
        assert "at least 1 character" in str(exc_info.value)

    def test_summary_too_long_fails(self):
        """Summary > 500 chars debe fallar."""
        with pytest.raises(ValidationError) as exc_info:
            AgentEvaluationOutput(score=50, summary="x" * 501, confidence_fields_present=True)
        assert "at most 500 characters" in str(exc_info.value)


class TestSpecificEvaluations:
    """Tests para schemas específicos por agente."""

    def test_feasibility_evaluation(self):
        eval = FeasibilityEvaluation(score=70, summary="Factible", confidence_fields_present=True)
        assert eval.score == 70

    def test_impact_evaluation(self):
        eval = ImpactEvaluation(score=80, summary="Alto impacto", confidence_fields_present=True)
        assert eval.score == 80

    def test_cost_evaluation_with_cost(self):
        eval = CostEvaluation(
            score=60,
            summary="Costo moderado",
            confidence_fields_present=True,
            costo_estimado_usd=1500000.0
        )
        assert eval.costo_estimado_usd == 1500000.0

    def test_cost_evaluation_without_cost(self):
        eval = CostEvaluation(score=60, summary="Costo moderado", confidence_fields_present=True)
        assert eval.costo_estimado_usd is None

    def test_cost_evaluation_negative_cost_fails(self):
        with pytest.raises(ValidationError) as exc_info:
            CostEvaluation(score=60, summary="Test", confidence_fields_present=True, costo_estimado_usd=-100)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_novelty_evaluation_defaults(self):
        eval = NoveltyEvaluation(score=75, summary="Novel", confidence_fields_present=True)
        assert eval.colision_patente_detectada is False
        assert eval.fuentes_consultadas == []

    def test_novelty_evaluation_with_patent_collision(self):
        eval = NoveltyEvaluation(
            score=40,
            summary="Posible colisión",
            confidence_fields_present=True,
            colision_patente_detectada=True,
            fuentes_consultadas=["USPTO", "EPO"]
        )
        assert eval.colision_patente_detectada is True
        assert eval.fuentes_consultadas == ["USPTO", "EPO"]


class TestRubric:
    """Tests para el schema de Rúbrica."""

    def test_valid_rubric(self):
        rubric = Rubric(
            id="test_1",
            criterio=Criterio.FEASIBILITY,
            campos_requeridos=["campo1"],
            campos_opcionales=["campo2"],
            criterios_de_evaluacion="Evalúa factibilidad",
            escala_guia={"0-50": "Bajo", "51-100": "Alto"}
        )
        assert rubric.criterio == Criterio.FEASIBILITY

    def test_rubric_empty_criterios_fails(self):
        with pytest.raises(ValidationError) as exc_info:
            Rubric(
                id="test",
                criterio=Criterio.IMPACT,
                criterios_de_evaluacion="",
            )
        assert "at least 1 character" in str(exc_info.value)


class TestRiskRule:
    """Tests para el schema de Regla de Riesgo."""

    def test_valid_risk_rule_gt(self):
        rule = RiskRule(
            id="risk_1",
            criterio=Criterio.COST,
            campo_a_evaluar="costo_estimado_usd",
            operador=OperadorRiesgo.GT,
            valor_umbral=1000000.0,
            descripcion_razon="Costo muy alto"
        )
        assert rule.operador == OperadorRiesgo.GT
        assert rule.valor_umbral == 1000000.0

    def test_valid_risk_rule_is_true(self):
        rule = RiskRule(
            id="risk_2",
            criterio=Criterio.NOVELTY,
            campo_a_evaluar="colision_patente_detectada",
            operador=OperadorRiesgo.IS_TRUE,
            valor_umbral=True,
            descripcion_razon="Colisión patente"
        )
        assert rule.valor_umbral is True


class TestCriteriaWeights:
    """Tests para pesos de criterios."""

    def test_default_weights(self):
        weights = CriteriaWeights()
        assert weights.feasibility == 0.25
        assert weights.impact == 0.25
        assert weights.cost == 0.25
        assert weights.novelty == 0.25

    def test_custom_valid_weights(self):
        weights = CriteriaWeights(feasibility=0.3, impact=0.3, cost=0.2, novelty=0.2)
        assert weights.feasibility == 0.3

    def test_invalid_weights_sum_fails(self):
        with pytest.raises(ValidationError) as exc_info:
            CriteriaWeights(feasibility=0.5, impact=0.5, cost=0.5, novelty=0.5)
        assert "sumar 1.0" in str(exc_info.value)


class TestEvaluationState:
    """Tests para el estado completo del grafo."""

    def test_minimal_state(self):
        state = EvaluationState(
            proposal_id="prop_123",
            proposal_text="Propuesta de prueba"
        )
        assert state.proposal_id == "prop_123"
        assert state.proposal_text == "Propuesta de prueba"
        assert state.feasibility_result is None
        assert state.hitl_required is False
        assert state.retry_count == 0

    def test_state_with_results(self):
        state = EvaluationState(
            proposal_id="prop_123",
            proposal_text="Test",
            feasibility_result=FeasibilityEvaluation(score=80, summary="OK", confidence_fields_present=True),
            weighted_score=75.5,
            hitl_required=True,
            hitl_reason="Costo alto"
        )
        assert state.feasibility_result.score == 80
        assert state.weighted_score == 75.5
        assert state.hitl_required is True
        assert state.hitl_reason == "Costo alto"

    def test_weighted_score_bounds(self):
        with pytest.raises(ValidationError):
            EvaluationState(proposal_id="1", proposal_text="t", weighted_score=-1)
        with pytest.raises(ValidationError):
            EvaluationState(proposal_id="1", proposal_text="t", weighted_score=101)


class TestEnums:
    """Tests para enums."""

    def test_criterio_values(self):
        assert Criterio.FEASIBILITY.value == "feasibility"
        assert Criterio.IMPACT.value == "impact"
        assert Criterio.COST.value == "cost"
        assert Criterio.NOVELTY.value == "novelty"

    def test_operador_riesgo_values(self):
        assert OperadorRiesgo.GT.value == "gt"
        assert OperadorRiesgo.LT.value == "lt"
        assert OperadorRiesgo.EQ.value == "eq"
        assert OperadorRiesgo.IS_TRUE.value == "is_true"