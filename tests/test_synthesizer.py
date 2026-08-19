import pytest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from proposal_evaluator.schemas import EvaluationState, FeasibilityEvaluation, ImpactEvaluation, CostEvaluation, NoveltyEvaluation
from proposal_evaluator.agents.synthesizer import (
    validate_numbers_in_narrative,
    extract_numbers,
    synthesizer_agent,
    build_final_report,
    _log_hallucination_detected,
)


# ─── Tests: Number Extraction ───
class TestExtractNumbers:
    def test_extract_integers(self):
        nums = extract_numbers("Score 80 y 90 son buenos")
        assert "80" in nums
        assert "90" in nums

    def test_extract_decimals(self):
        nums = extract_numbers("Costo 150000.50 y 77.75")
        assert "150000.50" in nums
        assert "77.75" in nums

    def test_extract_percentages(self):
        nums = extract_numbers("Mejora del 25% y 50.5%")
        assert "25%" in nums
        assert "50.5%" in nums

    def test_extract_currency(self):
        nums = extract_numbers("Costo $5000000 y USD 100000")
        assert "$5000000" in nums
        assert "USD 100000" in nums

    def test_extract_ranges(self):
        nums = extract_numbers("Rango 80-90 y 100-200")
        assert "80-90" in nums
        assert "100-200" in nums


# ─── Tests: Anti-Hallucination Validation ───
class TestAntiHallucinationValidation:
    def setup_method(self):
        self.allowed = {
            "feasibility_score": 80,
            "impact_score": 85,
            "cost_score": 70,
            "novelty_score": 75,
            "weighted_score": 77.5,
            "costo_estimado_usd": 150000.0,
        }

    def test_valid_narrative_no_numbers(self):
        narrative = "La propuesta tiene fortalezas significativas en innovación y presenta algunos riesgos técnicos moderados."
        is_valid, unrecognized = validate_numbers_in_narrative(narrative, self.allowed, "test_1")
        assert is_valid is True
        assert unrecognized == []

    def test_valid_narrative_with_allowed_numbers(self):
        narrative = "El score de factibilidad es 80 y el ponderado 77.5, lo cual es bueno."
        is_valid, unrecognized = validate_numbers_in_narrative(narrative, self.allowed, "test_2")
        assert is_valid is True
        assert unrecognized == []

    def test_invalid_narrative_with_hallucinated_number(self):
        narrative = "El score de factibilidad es 95 y el costo 9999999, muy alto."
        is_valid, unrecognized = validate_numbers_in_narrative(narrative, self.allowed, "test_3")
        assert is_valid is False
        assert "95" in unrecognized or "9999999" in unrecognized

    def test_invalid_narrative_with_percentage_not_in_allowed(self):
        narrative = "La mejora es del 99% respecto al baseline."
        is_valid, unrecognized = validate_numbers_in_narrative(narrative, self.allowed, "test_4")
        assert is_valid is False
        assert "99%" in unrecognized

    def test_invalid_narrative_with_currency_not_in_allowed(self):
        narrative = "El proyecto requiere una inversión de $9999999 inicial."
        is_valid, unrecognized = validate_numbers_in_narrative(narrative, self.allowed, "test_5")
        assert is_valid is False
        assert "$9999999" in unrecognized


# ─── Tests: Synthesizer Agent (with mocked LLM) ───
MOCK_STATE = EvaluationState(
    proposal_id="test_synth_1",
    proposal_text="Propuesta de prueba completa",
    raw_proposal_data={},
    feasibility_result=FeasibilityEvaluation(score=80, summary="OK", confidence_fields_present=True),
    impact_result=ImpactEvaluation(score=85, summary="OK", confidence_fields_present=True),
    cost_result=CostEvaluation(score=70, summary="OK", confidence_fields_present=True, costo_estimado_usd=150000.0),
    novelty_result=NoveltyEvaluation(score=75, summary="OK", confidence_fields_present=True, colision_patente_detectada=False, fuentes_consultadas=[]),
    weighted_score=77.5,
    hitl_required=False,
)


class TestSynthesizerAgent:
    @patch("proposal_evaluator.agents.synthesizer.call_structured")
    @patch("proposal_evaluator.agents.synthesizer.log_audit_event")
    def test_synthesizer_generates_narrative_without_hallucination(self, mock_log, mock_call):
        mock_call.return_value = type('obj', (object,), {
            'narrative': 'La propuesta presenta fortalezas técnicas claras y un enfoque innovador diferenciado. Los riesgos son manejables.'
        })()
        
        result = synthesizer_agent(MOCK_STATE)
        
        assert result.final_report is not None
        assert "77.5" in result.final_report  # weighted_score en el template
        assert "80" in result.final_report    # feasibility score
        assert "Fortalezas" in result.final_report or "fortalezas" in result.final_report.lower()
        assert result.requiere_revision_manual is False
        mock_call.assert_called_once()

    @patch("proposal_evaluator.agents.synthesizer.call_structured")
    @patch("proposal_evaluator.agents.synthesizer.log_audit_event")
    @patch("proposal_evaluator.agents.synthesizer.interrupt")
    def test_synthesizer_detects_hallucination_and_pauses(self, mock_interrupt, mock_log, mock_call):
        # LLM devuelve narrativa CON número alucinado (95 no está en allowed)
        mock_call.return_value = type('obj', (object,), {
            'narrative': 'El score de factibilidad es 95 y el costo 9999999, muy alto. Buena propuesta.'
        })()
        # interrupt debe ser mockado para evitar error fuera de contexto LangGraph
        mock_interrupt.return_value = {"action": "approve_text"}  # Simula decisión humana
        
        result = synthesizer_agent(MOCK_STATE)
        
        # Nueva comportamiento: PAUSA el grafo, NO ensambla reporte
        assert result.final_report is None  # NO ensambla reporte
        assert result.hitl_required is True
        assert result.hitl_type == "narrative_hallucination"
        assert "95" in result.hitl_reason or "9999999" in result.hitl_reason
        assert result.requiere_revision_manual is True
        mock_call.assert_called_once()
        mock_interrupt.assert_called_once()

    @patch("proposal_evaluator.agents.synthesizer.call_structured")
    @patch("proposal_evaluator.agents.synthesizer.log_audit_event")
    @patch("proposal_evaluator.agents.synthesizer.interrupt")
    def test_synthesizer_handles_llm_error(self, mock_interrupt, mock_log, mock_call):
        from proposal_evaluator.llm.client import LLMCallError
        from proposal_evaluator.schemas import EvaluationState, FeasibilityEvaluation, ImpactEvaluation, CostEvaluation, NoveltyEvaluation
        
        # Crear un estado FRESCO para este test (no reutilizar MOCK_STATE que puede estar contaminado)
        fresh_state = EvaluationState(
            proposal_id="test_llm_error",
            proposal_text="Propuesta de prueba para error LLM",
            raw_proposal_data={},
            feasibility_result=FeasibilityEvaluation(score=80, summary="OK", confidence_fields_present=True),
            impact_result=ImpactEvaluation(score=85, summary="OK", confidence_fields_present=True),
            cost_result=CostEvaluation(score=70, summary="OK", confidence_fields_present=True, costo_estimado_usd=150000.0),
            novelty_result=NoveltyEvaluation(score=75, summary="OK", confidence_fields_present=True, colision_patente_detectada=False, fuentes_consultadas=[]),
            weighted_score=77.5,
        )
        mock_call.side_effect = LLMCallError("API Error")
        
        result = synthesizer_agent(fresh_state)
        
        assert result.final_report is not None
        assert "Error" in result.final_report
        assert result.requiere_revision_manual is True

    @patch("proposal_evaluator.agents.synthesizer.call_structured")
    @patch("proposal_evaluator.agents.synthesizer.log_audit_event")
    @patch("proposal_evaluator.agents.synthesizer.interrupt")
    def test_narrative_hitl_approve_text(self, mock_interrupt, mock_log, mock_call):
        """Test: approve_text ensambla reporte con narrativa original y completa flujo."""
        from proposal_evaluator.schemas import EvaluationState, FeasibilityEvaluation, ImpactEvaluation, CostEvaluation, NoveltyEvaluation
        
        state = MOCK_STATE.model_copy(update={
            "hitl_required": True,
            "hitl_type": "narrative_hallucination",
            "hitl_reason": "Narrativa contiene cifras no verificadas: 95, 9999999",
            "hitl_decision": {
                "action": "approve_text",
                "qualitative_text": "La propuesta presenta fortalezas técnicas claras y un enfoque innovador diferenciado. Los riesgos son manejables.",
                "unrecognized_numbers": ["95", "9999999"],
            },
        })
        
        mock_interrupt.return_value = {"action": "approve_text"}
        
        result = synthesizer_agent(state)
        
        # approve_text debe ensamblar reporte con narrativa original
        assert result.final_report is not None
        assert "fortalezas técnicas claras" in result.final_report
        assert "77.5" in result.final_report  # weighted_score en template
        assert result.hitl_required is False
        assert result.hitl_type is None
        assert result.requiere_revision_manual is True

    @patch("proposal_evaluator.agents.synthesizer.call_structured")
    @patch("proposal_evaluator.agents.synthesizer.log_audit_event")
    @patch("proposal_evaluator.agents.synthesizer.interrupt")
    def test_narrative_hitl_regenerate(self, mock_interrupt, mock_log, mock_call):
        """Test: regenerate incrementa contador, llama sintetizador con corrección y continúa."""
        # Use a counter to track calls and return appropriate response
        call_count = {'count': 0}
        def mock_call_side_effect(*args, **kwargs):
            mock_call.call_count += 1
            if mock_call.call_count == 1:
                return type('obj', (object,), {'narrative': 'El score de factibilidad es 95 y el costo 9999999, muy alto.'})()
            else:
                return type('obj', (object,), {'narrative': 'La propuesta presenta fortalezas técnicas claras y un enfoque innovador. Los riesgos son manejables sin cifras prohibidas.'})()
        mock_call.side_effect = lambda *a, **kw: mock_call_side_effect(mock_call, *a, **kw)
        
        def mock_call_side_effect(mock_obj, *args, **kwargs):
            mock_obj.call_count += 1
            if mock_obj.call_count == 1:
                return type('obj', (object,), {'narrative': 'El score de factibilidad es 95 y el costo 9999999, muy alto.'})()
            else:
                return type('obj', (object,), {'narrative': 'La propuesta presenta fortalezas técnicas claras y un enfoque innovador. Los riesgos son manejables sin cifras prohibidas.'})()
        
        mock_call.side_effect = lambda *a, **kw: mock_call_side_effect(mock_call, *a, **kw)
        mock_call.call_count = 0
        
        # interrupt: first regenerate, second approve_text
        interrupt_count = [0]
        def mock_interrupt_func(payload):
            interrupt_count[0] += 1
            if interrupt_count[0] == 1:
                return {"action": "regenerate"}
            else:
                return {"action": "approve_text"}
        mock_interrupt.side_effect = mock_interrupt_func
        
        state = MOCK_STATE.model_copy(update={
            "hitl_required": True,
            "hitl_type": "narrative_hallucination",
            "hitl_reason": "Narrativa contiene cifras no verificadas: 95, 9999999",
            "hitl_decision": {
                "action": "regenerate",
                "qualitative_text": "El score de factibilidad es 95 y el costo 9999999, muy alto.",
                "unrecognized_numbers": ["95", "9999999"],
            },
            "narrative_retry_count": 0,
        })
        
        result = synthesizer_agent(state)
        
        # Debe haber incrementado contador y reintentado
        assert result.narrative_retry_count == 1
        assert result.final_report is not None
        assert "fortalezas técnicas claras" in result.final_report.lower()
        assert mock_call.call_count >= 2

    @patch("proposal_evaluator.agents.synthesizer.call_structured")
    @patch("proposal_evaluator.agents.synthesizer.log_audit_event")
    @patch("proposal_evaluator.agents.synthesizer.interrupt")
    def test_narrative_max_retries_exhausted(self, mock_interrupt, mock_log, mock_call):
        """Test: 3 regeneraciones fallidas agotan límite → exhausted."""
        # Mock: LLM siempre devuelve alucinación
        mock_call.return_value = type('obj', (object,), {
            'narrative': 'El score es 999 y el costo 9999999, muy alto.'
        })()
        # Mock interrupt: siempre pide regenerate
        mock_interrupt.return_value = {"action": "regenerate"}
        
        state = MOCK_STATE.model_copy(update={
            "hitl_required": True,
            "hitl_type": "narrative_hallucination",
            "hitl_reason": "Narrativa contiene cifras no verificadas",
            "hitl_decision": {"action": "regenerate", "unrecognized_numbers": ["999"]},
            "narrative_retry_count": 3,  # Ya en el límite
        })
        
        result = synthesizer_agent(state)
        
        assert result.hitl_type == "narrative_hallucination_exhausted"
        assert result.hitl_required is True
        assert "agotado" in result.hitl_reason.lower()
        assert result.final_report is None
        assert result.narrative_retry_count == 3

    @patch("proposal_evaluator.agents.synthesizer.call_structured")
    @patch("proposal_evaluator.agents.synthesizer.log_audit_event")
    @patch("proposal_evaluator.agents.synthesizer.interrupt")
    def test_synthesizer_handles_llm_error(self, mock_interrupt, mock_log, mock_call):
        from proposal_evaluator.llm.client import LLMCallError
        from proposal_evaluator.schemas import EvaluationState, FeasibilityEvaluation, ImpactEvaluation, CostEvaluation, NoveltyEvaluation
        
        # Crear un estado FRESCO para este test (no reutilizar MOCK_STATE que puede estar contaminado)
        fresh_state = EvaluationState(
            proposal_id="test_llm_error",
            proposal_text="Propuesta de prueba para error LLM",
            raw_proposal_data={},
            feasibility_result=FeasibilityEvaluation(score=80, summary="OK", confidence_fields_present=True),
            impact_result=ImpactEvaluation(score=85, summary="OK", confidence_fields_present=True),
            cost_result=CostEvaluation(score=70, summary="OK", confidence_fields_present=True, costo_estimado_usd=150000.0),
            novelty_result=NoveltyEvaluation(score=75, summary="OK", confidence_fields_present=True, colision_patente_detectada=False, fuentes_consultadas=[]),
            weighted_score=77.5,
        )
        mock_call.side_effect = LLMCallError("API Error")
        
        result = synthesizer_agent(fresh_state)
        
        assert result.final_report is not None
        assert "Error" in result.final_report
        assert result.requiere_revision_manual is True


# ─── Tests: Report Building ───
class TestReportBuilding:
    def test_report_contains_exact_scores(self):
        narrative = "Análisis cualitativo de la propuesta."
        report = build_final_report(MOCK_STATE, narrative)
        
        assert "77.5" in report       # weighted_score
        assert "80" in report         # feasibility
        assert "85" in report         # impact
        assert "70" in report         # cost
        assert "75" in report         # novelty
        assert "150,000.00" in report or "150000" in report  # costo
        assert narrative in report
        assert "REPORTE DE EVALUACIÓN" in report

    def test_report_includes_hitl_warning(self):
        state_hitl = MOCK_STATE.model_copy(update={"hitl_required": True, "hitl_reason": "Costo alto"})
        narrative = "Análisis."
        report = build_final_report(state_hitl, narrative)
        assert "HITL REQUERIDO:   SÍ" in report
        assert "Costo alto" in report


# ─── Tests: Audit Logging ───
class TestAuditLogging:
    @patch("proposal_evaluator.agents.synthesizer.log_audit_event")
    def test_audit_event_includes_number_context(self, mock_log):
        """Test: evento de auditoría incluye número no reconocido Y contexto ±80 chars."""
        from proposal_evaluator.agents.synthesizer import _log_hallucination_detected
        
        qualitative_text = "La propuesta tiene un score de factibilidad 95 y el costo 9999999 es muy alto para el presupuesto disponible. Se recomienda revisar."
        unrecognized = ["95", "9999999"]
        allowed_values = {"feasibility_score": 80, "costo_estimado_usd": 150000}
        
        _log_hallucination_detected("test_audit_1", unrecognized, qualitative_text, allowed_values)
        
        # Verificar que se llamó con el contexto correcto
        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert call_args[1]["event_type"] == "narrative_hallucination_detected"
        event_data = call_args[1]["event_data"]
        assert "context_snippets" in event_data
        assert len(event_data["context_snippets"]) == 2
        
        # Verificar contexto alrededor del 95
        snippet_95 = next(s for s in event_data["context_snippets"] if s["unrecognized_number"] == "95")
        assert "score de factibilidad 95" in snippet_95["surrounding_text"]
        assert len(snippet_95["surrounding_text"]) <= 160  # ±80 chars aprox
        
        # Verificar contexto alrededor del 9999999
        snippet_999 = next(s for s in event_data["context_snippets"] if s["unrecognized_number"] == "9999999")
        assert "costo 9999999" in snippet_999["surrounding_text"]