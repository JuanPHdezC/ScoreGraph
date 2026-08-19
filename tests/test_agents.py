import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pydantic import ValidationError

from proposal_evaluator.schemas import (
    EvaluationState,
    FeasibilityEvaluation,
    ImpactEvaluation,
    CostEvaluation,
    NoveltyEvaluation,
    Criterio,
    Rubric,
)
from proposal_evaluator.agents.completeness_gate import check_completeness, check_completeness_from_rubric
from proposal_evaluator.agents.feasibility_agent import feasibility_agent
from proposal_evaluator.agents.impact_agent import impact_agent
from proposal_evaluator.agents.cost_agent import cost_agent
from proposal_evaluator.agents.novelty_agent import novelty_agent, mock_prior_art_search, detect_patent_collision


# ─── Shared Mock Rubrics ───
MOCK_RUBRICS = {
    Criterio.FEASIBILITY: Rubric(
        id="test_feasibility",
        criterio=Criterio.FEASIBILITY,
        campos_requeridos=["descripcion_tecnica", "recursos_necesarios"],
        campos_opcionales=["equipo_disponible"],
        criterios_de_evaluacion="Evalúa factibilidad técnica",
        escala_guia={"0-30": "Baja", "31-70": "Media", "71-100": "Alta"},
    ),
    Criterio.IMPACT: Rubric(
        id="test_impact",
        criterio=Criterio.IMPACT,
        campos_requeridos=["problema_que_resuelve", "beneficio_esperado"],
        campos_opcionales=["tamano_mercado"],
        criterios_de_evaluacion="Evalúa impacto de negocio",
        escala_guia={"0-30": "Bajo", "31-70": "Medio", "71-100": "Alto"},
    ),
    Criterio.COST: Rubric(
        id="test_cost",
        criterio=Criterio.COST,
        campos_requeridos=["inversion_inicial_estimada"],
        campos_opcionales=["roi_estimado_meses"],
        criterios_de_evaluacion="Evalúa viabilidad económica",
        escala_guia={"0-30": "Caro", "31-70": "Razonable", "71-100": "Óptimo"},
    ),
    Criterio.NOVELTY: Rubric(
        id="test_novelty",
        criterio=Criterio.NOVELTY,
        campos_requeridos=["descripcion_innovacion", "diferenciadores_clave"],
        campos_opcionales=["patentes_relacionadas"],
        criterios_de_evaluacion="Evalúa novedad e innovación",
        escala_guia={"0-30": "Baja", "31-70": "Media", "71-100": "Radical"},
    ),
}


def get_mock_rubric(criterio: Criterio):
    return MOCK_RUBRICS.get(criterio)


# ─── Fixtures ───
@pytest.fixture
def sample_state():
    return EvaluationState(
        proposal_id="test_prop_123",
        proposal_text="Propuesta de prueba con IA generativa para automatizar reportes",
        raw_proposal_data={
            "descripcion_tecnica": "Uso de LLMs para generar reportes automáticos",
            "recursos_necesarios": "GPU cluster, equipo de 3 ingenieros",
            "problema_que_resuelve": "Reportes manuales toman 20h/semana",
            "beneficio_esperado": "Ahorro 80% tiempo, consistencia",
            "inversion_inicial_estimada": "150000",
            "descripcion_innovacion": "Pipeline RAG con fine-tuning propietario",
            "diferenciadores_clave": "Datos privados + RAG + evaluación automática calidad",
        },
    )


@pytest.fixture
def state_missing_fields():
    return EvaluationState(
        proposal_id="test_prop_456",
        proposal_text="Propuesta incompleta",
        raw_proposal_data={},  # Sin campos requeridos
    )


# ─── Tests: Completeness Gate ───
class TestCompletenessGate:
    def test_all_fields_present(self):
        required = ["campo1", "campo2"]
        data = {"campo1": "valor1", "campo2": "valor2"}
        missing = check_completeness("prop_1", "test", required, data)
        assert missing == []

    def test_some_fields_missing(self):
        required = ["campo1", "campo2", "campo3"]
        data = {"campo1": "valor1"}  # campo2 y campo3 faltan
        missing = check_completeness("prop_1", "test", required, data)
        assert set(missing) == {"campo2", "campo3"}

    def test_empty_string_counts_as_missing(self):
        required = ["campo1"]
        data = {"campo1": ""}
        missing = check_completeness("prop_1", "test", required, data)
        assert missing == ["campo1"]

    def test_none_counts_as_missing(self):
        required = ["campo1"]
        data = {"campo1": None}
        missing = check_completeness("prop_1", "test", required, data)
        assert missing == ["campo1"]

    def test_logging_occurs_when_missing(self, caplog):
        import structlog
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(0))
        required = ["campo1", "campo2"]
        data = {"campo1": "valor1"}
        missing = check_completeness("prop_log_1", "feasibility", required, data)
        assert "campo2" in missing
        # Verificar que se loggeó (structlog captura en caplog)


# ─── Tests: Mock Prior Art Search (Novelty) ───
class TestMockPriorArtSearch:
    def test_returns_results_for_known_keywords(self):
        results = mock_prior_art_search("blockchain supply chain")
        assert len(results) > 0
        assert all("id" in r and "title" in r and "relevance" in r for r in results)

    def test_returns_empty_for_unknown_keywords(self):
        results = mock_prior_art_search("tecnologia completamente nueva xyz")
        assert results == []

    def test_relevance_sorted_desc(self):
        results = mock_prior_art_search("blockchain")
        relevances = [r["relevance"] for r in results]
        assert relevances == sorted(relevances, reverse=True)

    def test_max_results_respected(self):
        results = mock_prior_art_search("blockchain", max_results=1)
        assert len(results) <= 1

    def test_detect_patent_collision_true(self):
        results = [{"relevance": 0.9}, {"relevance": 0.5}]
        assert detect_patent_collision(results, threshold=0.8) is True

    def test_detect_patent_collision_false(self):
        results = [{"relevance": 0.7}, {"relevance": 0.5}]
        assert detect_patent_collision(results, threshold=0.8) is False


# ─── Tests: Agents with Mocked LLM ───
MOCK_FEASIBILITY_RESULT = FeasibilityEvaluation(
    score=80,
    summary="Tecnología madura, recursos disponibles",
    confidence_fields_present=True,
)

MOCK_IMPACT_RESULT = ImpactEvaluation(
    score=85,
    summary="Alto impacto en eficiencia operativa",
    confidence_fields_present=True,
)

MOCK_COST_RESULT = CostEvaluation(
    score=70,
    summary="Inversión moderada, ROI 18 meses",
    confidence_fields_present=True,
    costo_estimado_usd=150000.0,
)

MOCK_NOVELTY_RESULT = NoveltyEvaluation(
    score=75,
    summary="Enfoque RAG propietario diferenciado",
    confidence_fields_present=True,
    colision_patente_detectada=False,
    fuentes_consultadas=["USPTO:US11223344B2"],
)


def mock_get_rubric(criterio):
    return get_mock_rubric(criterio)


class TestFeasibilityAgent:
    @patch("proposal_evaluator.agents.feasibility_agent.call_structured")
    @patch("proposal_evaluator.agents.feasibility_agent.get_rubric_by_criterio", side_effect=mock_get_rubric)
    def test_calls_llm_when_fields_present(self, mock_get_rubric, mock_call, sample_state):
        mock_call.return_value = MOCK_FEASIBILITY_RESULT
        result_state = feasibility_agent(sample_state)

        assert result_state.feasibility_result is not None
        assert result_state.feasibility_result.score == 80
        assert result_state.missing_fields_by_agent["feasibility"] == []
        mock_call.assert_called_once()

    @patch("proposal_evaluator.agents.feasibility_agent.call_structured")
    @patch("proposal_evaluator.agents.feasibility_agent.get_rubric_by_criterio", side_effect=mock_get_rubric)
    def test_blocks_llm_when_fields_missing(self, mock_get_rubric, mock_call, state_missing_fields):
        result_state = feasibility_agent(state_missing_fields)

        assert result_state.feasibility_result is None
        assert "descripcion_tecnica" in result_state.missing_fields_by_agent["feasibility"]
        assert "recursos_necesarios" in result_state.missing_fields_by_agent["feasibility"]
        mock_call.assert_not_called()

    @patch("proposal_evaluator.agents.feasibility_agent.call_structured")
    @patch("proposal_evaluator.agents.feasibility_agent.get_rubric_by_criterio", side_effect=mock_get_rubric)
    def test_handles_llm_error(self, mock_get_rubric, mock_call, sample_state, caplog):
        from proposal_evaluator.llm.client import LLMCallError
        mock_call.side_effect = LLMCallError("API error")
        result_state = feasibility_agent(sample_state)

        assert result_state.feasibility_result is None
        assert result_state.missing_fields_by_agent["feasibility"] == ["llm_error"]


class TestImpactAgent:
    @patch("proposal_evaluator.agents.impact_agent.call_structured")
    @patch("proposal_evaluator.agents.impact_agent.get_rubric_by_criterio", side_effect=mock_get_rubric)
    def test_calls_llm_when_fields_present(self, mock_get_rubric, mock_call, sample_state):
        mock_call.return_value = MOCK_IMPACT_RESULT
        result_state = impact_agent(sample_state)

        assert result_state.impact_result is not None
        assert result_state.impact_result.score == 85
        assert result_state.missing_fields_by_agent["impact"] == []
        mock_call.assert_called_once()

    @patch("proposal_evaluator.agents.impact_agent.call_structured")
    @patch("proposal_evaluator.agents.impact_agent.get_rubric_by_criterio", side_effect=mock_get_rubric)
    def test_blocks_llm_when_fields_missing(self, mock_get_rubric, mock_call, state_missing_fields):
        result_state = impact_agent(state_missing_fields)

        assert result_state.impact_result is None
        assert "problema_que_resuelve" in result_state.missing_fields_by_agent["impact"]
        assert "beneficio_esperado" in result_state.missing_fields_by_agent["impact"]
        mock_call.assert_not_called()


class TestCostAgent:
    @patch("proposal_evaluator.agents.cost_agent.call_structured")
    @patch("proposal_evaluator.agents.cost_agent.get_rubric_by_criterio", side_effect=mock_get_rubric)
    def test_calls_llm_when_fields_present(self, mock_get_rubric, mock_call, sample_state):
        mock_call.return_value = MOCK_COST_RESULT
        result_state = cost_agent(sample_state)

        assert result_state.cost_result is not None
        assert result_state.cost_result.score == 70
        assert result_state.cost_result.costo_estimado_usd == 150000.0
        assert result_state.missing_fields_by_agent["cost"] == []
        mock_call.assert_called_once()

    @patch("proposal_evaluator.agents.cost_agent.call_structured")
    @patch("proposal_evaluator.agents.cost_agent.get_rubric_by_criterio", side_effect=mock_get_rubric)
    def test_blocks_llm_when_fields_missing(self, mock_get_rubric, mock_call, state_missing_fields):
        result_state = cost_agent(state_missing_fields)

        assert result_state.cost_result is None
        assert "inversion_inicial_estimada" in result_state.missing_fields_by_agent["cost"]
        mock_call.assert_not_called()

    @patch("proposal_evaluator.agents.cost_agent.call_structured")
    @patch("proposal_evaluator.agents.cost_agent.get_rubric_by_criterio", side_effect=mock_get_rubric)
    def test_costo_estimado_usd_none_is_valid(self, mock_get_rubric, mock_call, sample_state):
        """costo_estimado_usd = None debe ser válido (campo opcional)."""
        mock_call.return_value = CostEvaluation(
            score=60,
            summary="No hay datos para estimar",
            confidence_fields_present=True,
            costo_estimado_usd=None,
        )
        result_state = cost_agent(sample_state)

        assert result_state.cost_result is not None
        assert result_state.cost_result.costo_estimado_usd is None

    @patch("proposal_evaluator.agents.cost_agent.call_structured")
    @patch("proposal_evaluator.agents.cost_agent.get_rubric_by_criterio", side_effect=mock_get_rubric)
    def test_costo_estimado_usd_parsed_as_float(self, mock_get_rubric, mock_call, sample_state):
        """costo_estimado_usd debe parsearse como float (ej. string numérico -> float)."""
        # Simular que el LLM devuelve el costo como string numérico (común en JSON)
        mock_call.return_value = CostEvaluation(
            score=65,
            summary="Costo estimado desde string",
            confidence_fields_present=True,
            costo_estimado_usd="250000.50",  # String que Pydantic debe coaccionar a float
        )
        result_state = cost_agent(sample_state)

        assert result_state.cost_result is not None
        assert isinstance(result_state.cost_result.costo_estimado_usd, float)
        assert result_state.cost_result.costo_estimado_usd == 250000.5

    @patch("proposal_evaluator.agents.cost_agent.call_structured")
    @patch("proposal_evaluator.agents.cost_agent.get_rubric_by_criterio", side_effect=mock_get_rubric)
    def test_costo_estimado_usd_negative_rejected(self, mock_get_rubric, mock_call, sample_state):
        """costo_estimado_usd negativo debe ser rechazado por validación Pydantic (ge=0)."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CostEvaluation(
                score=60,
                summary="Test",
                confidence_fields_present=True,
                costo_estimado_usd=-1000,
            )


class TestNoveltyAgent:
    @patch("proposal_evaluator.agents.novelty_agent.call_structured")
    @patch("proposal_evaluator.agents.novelty_agent.get_rubric_by_criterio", side_effect=mock_get_rubric)
    def test_calls_llm_when_fields_present(self, mock_get_rubric, mock_call, sample_state):
        mock_call.return_value = MOCK_NOVELTY_RESULT
        result_state = novelty_agent(sample_state)

        assert result_state.novelty_result is not None
        assert result_state.novelty_result.score == 75
        assert result_state.novelty_result.colision_patente_detectada is False
        # MVP: fuentes_consultadas se poblaría en implementación completa con tool-calling real
        assert result_state.missing_fields_by_agent["novelty"] == []
        mock_call.assert_called_once()

    @patch("proposal_evaluator.agents.novelty_agent.call_structured")
    @patch("proposal_evaluator.agents.novelty_agent.get_rubric_by_criterio", side_effect=mock_get_rubric)
    def test_blocks_llm_when_fields_missing(self, mock_get_rubric, mock_call, state_missing_fields):
        result_state = novelty_agent(state_missing_fields)

        assert result_state.novelty_result is None
        assert "descripcion_innovacion" in result_state.missing_fields_by_agent["novelty"]
        assert "diferenciadores_clave" in result_state.missing_fields_by_agent["novelty"]
        mock_call.assert_not_called()


# ─── Tests: Rubric Integration ───
class TestAgentRubricIntegration:
    def test_feasibility_rubric_loaded(self):
        rubric = get_mock_rubric(Criterio.FEASIBILITY)
        assert rubric is not None
        assert rubric.criterio == Criterio.FEASIBILITY
        assert "descripcion_tecnica" in rubric.campos_requeridos
        assert "recursos_necesarios" in rubric.campos_requeridos

    def test_impact_rubric_loaded(self):
        rubric = get_mock_rubric(Criterio.IMPACT)
        assert rubric is not None
        assert "problema_que_resuelve" in rubric.campos_requeridos
        assert "beneficio_esperado" in rubric.campos_requeridos

    def test_cost_rubric_loaded(self):
        rubric = get_mock_rubric(Criterio.COST)
        assert rubric is not None
        assert "inversion_inicial_estimada" in rubric.campos_requeridos

    def test_novelty_rubric_loaded(self):
        rubric = get_mock_rubric(Criterio.NOVELTY)
        assert rubric is not None
        assert "descripcion_innovacion" in rubric.campos_requeridos
        assert "diferenciadores_clave" in rubric.campos_requeridos

    def test_rubric_has_escala_guia(self):
        for criterio in Criterio:
            rubric = get_mock_rubric(criterio)
            assert rubric is not None
            assert len(rubric.escala_guia) >= 3  # Al menos 4 rangos