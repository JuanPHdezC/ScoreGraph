import pytest
from unittest.mock import patch, MagicMock
from proposal_evaluator.schemas import (
    EvaluationState,
    FeasibilityEvaluation,
    ImpactEvaluation,
    CostEvaluation,
    NoveltyEvaluation,
    Criterio,
    Rubric,
)
from proposal_evaluator.config.config_repository import get_criteria_weights
from proposal_evaluator.agents.aggregator import calculate_weighted_score
from proposal_evaluator.agents.risk_gate import evaluate_risk_rules
from proposal_evaluator.agents.missing_info_node import (
    missing_info_node,
    should_request_missing_info,
    build_missing_info_payload,
)
from proposal_evaluator.graph.build_graph import (
    build_graph,
    run_evaluation,
    resume_evaluation,
    get_evaluation_state,
)


# ─── Shared Mocks ───
MOCK_RUBRICS = {
    Criterio.FEASIBILITY: Rubric(
        id="test_feasibility",
        criterio=Criterio.FEASIBILITY,
        campos_requeridos=["descripcion_tecnica", "recursos_necesarios"],
        campos_opcionales=[],
        criterios_de_evaluacion="Test",
        escala_guia={"0-30": "Baja", "31-70": "Media", "71-100": "Alta"},
    ),
    Criterio.IMPACT: Rubric(
        id="test_impact",
        criterio=Criterio.IMPACT,
        campos_requeridos=["problema_que_resuelve", "beneficio_esperado"],
        campos_opcionales=[],
        criterios_de_evaluacion="Test",
        escala_guia={"0-30": "Baja", "31-70": "Media", "71-100": "Alta"},
    ),
    Criterio.COST: Rubric(
        id="test_cost",
        criterio=Criterio.COST,
        campos_requeridos=["inversion_inicial_estimada"],
        campos_opcionales=[],
        criterios_de_evaluacion="Test",
        escala_guia={"0-30": "Alto", "31-70": "Medio", "71-100": "Bajo"},
    ),
    Criterio.NOVELTY: Rubric(
        id="test_novelty",
        criterio=Criterio.NOVELTY,
        campos_requeridos=["descripcion_innovacion", "diferenciadores_clave"],
        campos_opcionales=[],
        criterios_de_evaluacion="Test",
        escala_guia={"0-30": "Baja", "31-70": "Media", "71-100": "Alta"},
    ),
}


MOCK_FEASIBILITY = FeasibilityEvaluation(score=80, summary="OK", confidence_fields_present=True)
MOCK_IMPACT = ImpactEvaluation(score=85, summary="OK", confidence_fields_present=True)
MOCK_COST = CostEvaluation(score=70, summary="OK", confidence_fields_present=True, costo_estimado_usd=150000.0)
MOCK_NOVELTY = NoveltyEvaluation(score=75, summary="OK", confidence_fields_present=True, colision_patente_detectada=False, fuentes_consultadas=[])

# Mock weights object
from proposal_evaluator.schemas import CriteriaWeights
MOCK_WEIGHTS = CriteriaWeights(feasibility=0.25, impact=0.25, cost=0.25, novelty=0.25)


def get_mock_rubric(criterio):
    return MOCK_RUBRICS.get(criterio)


# ─── Tests: Aggregator ───
class TestAggregator:
    @patch("proposal_evaluator.agents.aggregator.get_criteria_weights", return_value=MOCK_WEIGHTS)
    @patch("proposal_evaluator.agents.aggregator.log_audit_event")
    def test_calculates_weighted_score_correctly(self, mock_log, mock_weights):
        state = EvaluationState(
            proposal_id="test_1",
            proposal_text="Test",
            feasibility_result=MOCK_FEASIBILITY,
            impact_result=MOCK_IMPACT,
            cost_result=MOCK_COST,
            novelty_result=MOCK_NOVELTY,
        )
        result = calculate_weighted_score(state)

        # 80*0.25 + 85*0.25 + 70*0.25 + 75*0.25 = 20 + 21.25 + 17.5 + 18.75 = 77.5
        assert result.weighted_score == 77.5
        mock_log.assert_called_once()
        call_args = mock_log.call_args[1]["event_data"]
        assert call_args["weighted_score"] == 77.5
        assert call_args["individual_scores"]["feasibility"] == 80

    @patch("proposal_evaluator.agents.aggregator.get_criteria_weights", return_value=MOCK_WEIGHTS)
    def test_returns_none_if_missing_results(self, mock_weights):
        state = EvaluationState(
            proposal_id="test_2",
            proposal_text="Test",
            feasibility_result=MOCK_FEASIBILITY,
            impact_result=None,  # Faltante
            cost_result=MOCK_COST,
            novelty_result=MOCK_NOVELTY,
        )
        result = calculate_weighted_score(state)
        assert result.weighted_score is None


# ─── Tests: Risk Gate ───
class TestRiskGate:
    @patch("proposal_evaluator.agents.risk_gate.get_active_risk_rules")
    @patch("proposal_evaluator.agents.risk_gate.log_audit_event")
    def test_no_rules_triggered(self, mock_log, mock_rules):
        # Regla que NO se dispara (costo bajo)
        from proposal_evaluator.schemas import RiskRule, OperadorRiesgo
        mock_rules.return_value = [
            RiskRule(
                id="risk_cost_high",
                criterio=Criterio.COST,
                campo_a_evaluar="costo_estimado_usd",
                operador=OperadorRiesgo.GT,
                valor_umbral=5000000.0,
                descripcion_razon="Costo alto",
            ),
        ]

        state = EvaluationState(
            proposal_id="test_3",
            proposal_text="Test",
            cost_result=MOCK_COST,  # 150k < 5M
        )
        result = evaluate_risk_rules(state)

        assert result.hitl_required is False
        assert result.hitl_reason is None

    @patch("proposal_evaluator.agents.risk_gate.get_active_risk_rules")
    @patch("proposal_evaluator.agents.risk_gate.log_audit_event")
    def test_rule_triggered_sets_hitl(self, mock_log, mock_rules):
        # Regla que SÍ se dispara (costo alto)
        from proposal_evaluator.schemas import RiskRule, OperadorRiesgo
        mock_rules.return_value = [
            RiskRule(
                id="risk_cost_high",
                criterio=Criterio.COST,
                campo_a_evaluar="costo_estimado_usd",
                operador=OperadorRiesgo.GT,
                valor_umbral=100000.0,  # 100k threshold
                descripcion_razon="Costo supera 100k",
            ),
        ]

        state = EvaluationState(
            proposal_id="test_4",
            proposal_text="Test",
            cost_result=CostEvaluation(score=70, summary="OK", confidence_fields_present=True, costo_estimado_usd=150000.0),
        )
        result = evaluate_risk_rules(state)

        assert result.hitl_required is True
        assert "Costo supera 100k" in result.hitl_reason
        mock_log.assert_called()
        # Verificar que se loggeó hitl_triggered
        hitl_calls = [c for c in mock_log.call_args_list if c[1]["event_type"] == "hitl_triggered"]
        assert len(hitl_calls) == 1

    @patch("proposal_evaluator.agents.risk_gate.get_active_risk_rules")
    @patch("proposal_evaluator.agents.risk_gate.log_audit_event")
    def test_feasibility_low_score_triggers(self, mock_log, mock_rules):
        from proposal_evaluator.schemas import RiskRule, OperadorRiesgo
        mock_rules.return_value = [
            RiskRule(
                id="risk_feasibility_low",
                criterio=Criterio.FEASIBILITY,
                campo_a_evaluar="score",
                operador=OperadorRiesgo.LT,
                valor_umbral=50,
                descripcion_razon="Factibilidad baja",
            ),
        ]

        state = EvaluationState(
            proposal_id="test_5",
            proposal_text="Test",
            feasibility_result=FeasibilityEvaluation(score=30, summary="Baja", confidence_fields_present=True),
        )
        result = evaluate_risk_rules(state)

        assert result.hitl_required is True
        assert "Factibilidad baja" in result.hitl_reason


# ─── Tests: Missing Info Node ───
class TestMissingInfoNode:
    @patch("proposal_evaluator.agents.missing_info_node.log_audit_event")
    def test_continues_if_no_missing(self, mock_log):
        state = EvaluationState(
            proposal_id="test_6",
            proposal_text="Test",
            missing_fields_by_agent={},
        )
        result = missing_info_node(state)
        assert result.missing_fields_by_agent == {}
        assert result.retry_count == 0

    @patch("proposal_evaluator.agents.missing_info_node.log_audit_event")
    @patch("proposal_evaluator.agents.missing_info_node.interrupt", return_value=None)
    def test_requests_info_and_increments_retry(self, mock_interrupt, mock_log):
        state = EvaluationState(
            proposal_id="test_7",
            proposal_text="Test",
            missing_fields_by_agent={"feasibility": ["descripcion_tecnica"]},
            retry_count=0,
        )
        result = missing_info_node(state)

        assert result.retry_count == 1
        assert result.missing_fields_by_agent == {}  # Se limpia para re-evaluar

    @patch("proposal_evaluator.agents.missing_info_node.log_audit_event")
    @patch("proposal_evaluator.agents.missing_info_node.interrupt", return_value=None)
    def test_max_retries_forces_hitl(self, mock_interrupt, mock_log):
        state = EvaluationState(
            proposal_id="test_8",
            proposal_text="Test",
            missing_fields_by_agent={"feasibility": ["descripcion_tecnica"]},
            retry_count=3,  # Ya en MAX_RETRIES
        )
        result = missing_info_node(state)

        assert result.hitl_required is True
        assert result.hitl_reason == "info_incompleta_tras_reintentos"

    @patch("proposal_evaluator.agents.missing_info_node.log_audit_event")
    @patch("proposal_evaluator.agents.missing_info_node.interrupt", return_value={
        "raw_proposal_data": {"descripcion_tecnica": "Nueva info técnica"}
    })
    def test_resume_with_new_data_updates_raw_data(self, mock_interrupt, mock_log):
        state = EvaluationState(
            proposal_id="test_9",
            proposal_text="Test",
            missing_fields_by_agent={"feasibility": ["descripcion_tecnica"]},
            raw_proposal_data={},
            retry_count=0,
        )
        result = missing_info_node(state)

        assert "descripcion_tecnica" in result.raw_proposal_data
        assert result.raw_proposal_data["descripcion_tecnica"] == "Nueva info técnica"

    def test_should_request_missing_info_decision(self):
        state_with = EvaluationState(proposal_id="x", proposal_text="t", missing_fields_by_agent={"f": ["c"]})
        state_without = EvaluationState(proposal_id="x", proposal_text="t", missing_fields_by_agent={})
        assert should_request_missing_info(state_with) == "request_info"
        assert should_request_missing_info(state_without) == "continue"

    def test_build_missing_info_payload(self):
        state = EvaluationState(
            proposal_id="prop_123",
            proposal_text="Test",
            missing_fields_by_agent={"feasibility": ["a"], "cost": ["b"]},
            retry_count=1,
        )
        payload = build_missing_info_payload(state)
        assert payload["proposal_id"] == "prop_123"
        assert payload["missing_fields_by_agent"] == {"feasibility": ["a"], "cost": ["b"]}
        assert payload["retry_count"] == 1


# ─── Tests: Graph Integration (Full End-to-End) ───
class TestGraphIntegration:
    @patch("proposal_evaluator.agents.feasibility_agent.get_rubric_by_criterio", side_effect=get_mock_rubric)
    @patch("proposal_evaluator.agents.impact_agent.get_rubric_by_criterio", side_effect=get_mock_rubric)
    @patch("proposal_evaluator.agents.cost_agent.get_rubric_by_criterio", side_effect=get_mock_rubric)
    @patch("proposal_evaluator.agents.novelty_agent.get_rubric_by_criterio", side_effect=get_mock_rubric)
    @patch("proposal_evaluator.agents.feasibility_agent.call_structured", return_value=MOCK_FEASIBILITY)
    @patch("proposal_evaluator.agents.impact_agent.call_structured", return_value=MOCK_IMPACT)
    @patch("proposal_evaluator.agents.cost_agent.call_structured", return_value=MOCK_COST)
    @patch("proposal_evaluator.agents.novelty_agent.call_structured", return_value=MOCK_NOVELTY)
    @patch("proposal_evaluator.agents.risk_gate.get_active_risk_rules", return_value=[])
    @patch("proposal_evaluator.agents.aggregator.get_criteria_weights", return_value=MOCK_WEIGHTS)
    @patch("proposal_evaluator.graph.build_graph.log_audit_event")
    @patch("proposal_evaluator.agents.aggregator.log_audit_event")
    @patch("proposal_evaluator.agents.risk_gate.log_audit_event")
    @patch("proposal_evaluator.agents.missing_info_node.log_audit_event")
    def test_full_graph_complete_no_pause(
        self,
        mock_missing_log,
        mock_risk_log,
        mock_agg_log,
        mock_graph_log,
        mock_agg_weights,
        mock_risk_rules,
        mock_novelty_llm,
        mock_cost_llm,
        mock_impact_llm,
        mock_feas_llm,
        mock_novelty_rubric,
        mock_cost_rubric,
        mock_impact_rubric,
        mock_feas_rubric,
    ):
        """
        Test de integración: propuesta con toda la info y sin riesgo.
        Debe llegar hasta el sintetizador sin pausas.
        """
        state = run_evaluation(
            proposal_id="integ_1",
            proposal_text="Propuesta completa con todos los datos",
            raw_proposal_data={
                "descripcion_tecnica": "Tech details",
                "recursos_necesarios": "Resources",
                "problema_que_resuelve": "Problem",
                "beneficio_esperado": "Benefit",
                "inversion_inicial_estimada": "100000",
                "descripcion_innovacion": "Innovation",
                "diferenciadores_clave": "Differentiators",
            },
            use_test_graph=True,
        )

        # Verificar resultados finales
        assert state.feasibility_result is not None
        assert state.impact_result is not None
        assert state.cost_result is not None
        assert state.novelty_result is not None
        assert state.weighted_score is not None
        assert state.hitl_required is False
        assert state.final_report is not None

    @patch("proposal_evaluator.agents.feasibility_agent.get_rubric_by_criterio", side_effect=get_mock_rubric)
    @patch("proposal_evaluator.agents.impact_agent.get_rubric_by_criterio", side_effect=get_mock_rubric)
    @patch("proposal_evaluator.agents.cost_agent.get_rubric_by_criterio", side_effect=get_mock_rubric)
    @patch("proposal_evaluator.agents.novelty_agent.get_rubric_by_criterio", side_effect=get_mock_rubric)
    @patch("proposal_evaluator.agents.feasibility_agent.call_structured", return_value=MOCK_FEASIBILITY)
    @patch("proposal_evaluator.agents.impact_agent.call_structured", return_value=MOCK_IMPACT)
    @patch("proposal_evaluator.agents.cost_agent.call_structured", return_value=MOCK_COST)
    @patch("proposal_evaluator.agents.novelty_agent.call_structured", return_value=MOCK_NOVELTY)
    @patch("proposal_evaluator.agents.risk_gate.get_active_risk_rules", return_value=[])
    @patch("proposal_evaluator.agents.aggregator.get_criteria_weights", return_value=MOCK_WEIGHTS)
    @patch("proposal_evaluator.graph.build_graph.log_audit_event")
    @patch("proposal_evaluator.agents.aggregator.log_audit_event")
    @patch("proposal_evaluator.agents.risk_gate.log_audit_event")
    @patch("proposal_evaluator.agents.missing_info_node.log_audit_event")
    def test_graph_pauses_on_missing_info_and_resumes(
        self,
        mock_missing_log,
        mock_risk_log,
        mock_agg_log,
        mock_graph_log,
        mock_agg_weights,
        mock_risk_rules,
        mock_novelty_llm,
        mock_cost_llm,
        mock_impact_llm,
        mock_feas_llm,
        mock_novelty_rubric,
        mock_cost_rubric,
        mock_impact_rubric,
        mock_feas_rubric,
    ):
        """
        Test: propuesta con campos faltantes -> pausa -> reanuda con nuevos datos.
        """
        # Primera ejecución: faltan SOLO campos de factibilidad
        # Incluimos TODOS los campos requeridos de impact, cost, novelty
        state = run_evaluation(
            proposal_id="integ_2",
            proposal_text="Propuesta incompleta",
            raw_proposal_data={
                # Falta: descripcion_tecnica, recursos_necesarios (feasibility)
                "problema_que_resuelve": "Problem",
                "beneficio_esperado": "Benefit",
                "inversion_inicial_estimada": "100000",
                "descripcion_innovacion": "Innovation",
                "diferenciadores_clave": "Differentiators",
                # Campos requeridos para impact
                "tamano_mercado": "1000000",
                "usuarios_afectados": "1000",
                "ventaja_competitiva": "Unique",
                "kpis_objetivo": "ROI 20%",
                # Campos requeridos para cost
                "costo_operativo_anual": "50000",
                "roi_estimado_meses": "12",
                "fuentes_financiacion": "Internal",
                "desglose_costos": "Details",
                # Campos requeridos para novelty
                "patentes_relacionadas": "None",
                "estado_del_arte": "Current",
                "busqueda_previa_realizada": "Yes",
            },
            use_test_graph=True,
        )

        # Debe pausarse en missing_info (interrupt)
        # Verificar que se detectaron campos faltantes
        assert "descripcion_tecnica" in state.missing_fields_by_agent.get("feasibility", [])

        # Reanudar con datos faltantes + datos originales (para simular persistencia completa)
        resume_data = {
            "raw_proposal_data": {
                # Nuevos campos de feasibility
                "descripcion_tecnica": "Added later",
                "recursos_necesarios": "Resources added",
                # Campos originales de impact
                "problema_que_resuelve": "Problem",
                "beneficio_esperado": "Benefit",
                "tamano_mercado": "1000000",
                "usuarios_afectados": "1000",
                "ventaja_competitiva": "Unique",
                "kpis_objetivo": "ROI 20%",
                # Campos originales de cost
                "inversion_inicial_estimada": "100000",
                "costo_operativo_anual": "50000",
                "roi_estimado_meses": "12",
                "fuentes_financiacion": "Internal",
                "desglose_costos": "Details",
                # Campos originales de novelty
                "descripcion_innovacion": "Innovation",
                "diferenciadores_clave": "Differentiators",
                "patentes_relacionadas": "None",
                "estado_del_arte": "Current",
                "busqueda_previa_realizada": "Yes",
            }
        }
        state = resume_evaluation("integ_2", resume_data, use_test_graph=True)

        # Ahora debe completar sin pausas
        assert state.feasibility_result is not None
        assert state.weighted_score is not None

    @patch("proposal_evaluator.agents.feasibility_agent.get_rubric_by_criterio", side_effect=get_mock_rubric)
    @patch("proposal_evaluator.agents.impact_agent.get_rubric_by_criterio", side_effect=get_mock_rubric)
    @patch("proposal_evaluator.agents.cost_agent.get_rubric_by_criterio", side_effect=get_mock_rubric)
    @patch("proposal_evaluator.agents.novelty_agent.get_rubric_by_criterio", side_effect=get_mock_rubric)
    @patch("proposal_evaluator.agents.feasibility_agent.call_structured", return_value=MOCK_FEASIBILITY)
    @patch("proposal_evaluator.agents.impact_agent.call_structured", return_value=MOCK_IMPACT)
    @patch("proposal_evaluator.agents.cost_agent.call_structured", return_value=CostEvaluation(score=70, summary="OK", confidence_fields_present=True, costo_estimado_usd=6000000.0))  # > 5M
    @patch("proposal_evaluator.agents.novelty_agent.call_structured", return_value=MOCK_NOVELTY)
    @patch("proposal_evaluator.agents.risk_gate.get_active_risk_rules")
    @patch("proposal_evaluator.agents.aggregator.get_criteria_weights", return_value=MOCK_WEIGHTS)
    @patch("proposal_evaluator.graph.build_graph.log_audit_event")
    @patch("proposal_evaluator.agents.aggregator.log_audit_event")
    @patch("proposal_evaluator.agents.risk_gate.log_audit_event")
    @patch("proposal_evaluator.agents.missing_info_node.log_audit_event")
    def test_graph_triggers_hitl_on_risk_rule(
            self,
            mock_missing_log,
            mock_risk_log,
            mock_agg_log,
            mock_graph_log,
            mock_agg_weights,
            mock_risk_rules,
            mock_novelty_llm,
            mock_cost_llm,
            mock_impact_llm,
            mock_feas_llm,
            mock_novelty_rubric,
            mock_cost_rubric,
            mock_impact_rubric,
            mock_feas_rubric,
        ):
            """
            Test: regla de riesgo costo > 5M dispara HITL.
            """
            # Regla de costo alto activa
            from proposal_evaluator.schemas import RiskRule, OperadorRiesgo
            mock_risk_rules.return_value = [
                RiskRule(
                    id="risk_cost_high",
                    criterio=Criterio.COST,
                    campo_a_evaluar="costo_estimado_usd",
                    operador=OperadorRiesgo.GT,
                    valor_umbral=5000000.0,
                    descripcion_razon="Costo estimado superior a $5M requiere validación ejecutiva",
                ),
            ]
    
            state = run_evaluation(
                proposal_id="integ_3",
                proposal_text="Propuesta costosa",
                raw_proposal_data={
                    "descripcion_tecnica": "Tech",
                    "recursos_necesarios": "Resources",
                    "problema_que_resuelve": "Problem",
                    "beneficio_esperado": "Benefit",
                    "inversion_inicial_estimada": "6000000",
                    "descripcion_innovacion": "Innovation",
                    "diferenciadores_clave": "Diff",
                },
                use_test_graph=True,
            )
    
            # Debe pausarse en hitl_pause con hitl_required=True
            assert state.hitl_required is True
            assert "Costo estimado superior a $5M" in state.hitl_reason
            assert state.weighted_score is not None  # Se calculó antes del risk gate


# ─── Test de paralelismo real ───
class TestParallelExecution:
    """Test que verifica que el fan-out es realmente paralelo (no secuencial)."""

    @patch("proposal_evaluator.agents.feasibility_agent.get_rubric_by_criterio", side_effect=get_mock_rubric)
    @patch("proposal_evaluator.agents.impact_agent.get_rubric_by_criterio", side_effect=get_mock_rubric)
    @patch("proposal_evaluator.agents.cost_agent.get_rubric_by_criterio", side_effect=get_mock_rubric)
    @patch("proposal_evaluator.agents.novelty_agent.get_rubric_by_criterio", side_effect=get_mock_rubric)
    @patch("proposal_evaluator.agents.feasibility_agent.call_structured")
    @patch("proposal_evaluator.agents.impact_agent.call_structured")
    @patch("proposal_evaluator.agents.cost_agent.call_structured")
    @patch("proposal_evaluator.agents.novelty_agent.call_structured")
    @patch("proposal_evaluator.agents.risk_gate.get_active_risk_rules", return_value=[])
    @patch("proposal_evaluator.agents.aggregator.get_criteria_weights", return_value=MOCK_WEIGHTS)
    @patch("proposal_evaluator.graph.build_graph.log_audit_event")
    @patch("proposal_evaluator.agents.aggregator.log_audit_event")
    @patch("proposal_evaluator.agents.risk_gate.log_audit_event")
    @patch("proposal_evaluator.agents.missing_info_node.log_audit_event")
    def test_fan_out_is_parallel_not_sequential(
        self,
        mock_missing_log,
        mock_risk_log,
        mock_agg_log,
        mock_graph_log,
        mock_agg_weights,
        mock_risk_rules,
        mock_novelty_llm,
        mock_cost_llm,
        mock_impact_llm,
        mock_feas_llm,
        mock_novelty_rubric,
        mock_cost_rubric,
        mock_impact_rubric,
        mock_feas_rubric,
    ):
        """
        Verifica que los 4 agentes se ejecutan EN PARALELO, no en cadena.
        
        Cada mock de LLM tiene un delay artificial de 0.1s.
        Si fueran secuenciales: tiempo total >= 0.4s (4 * 0.1s).
        Si son paralelos: tiempo total ≈ 0.1s (el más lento).
        """
        import time
        
        # Configurar mocks con delay artificial
        def slow_feasibility(*args, **kwargs):
            time.sleep(0.1)
            return MOCK_FEASIBILITY
        
        def slow_impact(*args, **kwargs):
            time.sleep(0.1)
            return MOCK_IMPACT
        
        def slow_cost(*args, **kwargs):
            time.sleep(0.1)
            return MOCK_COST
        
        def slow_novelty(*args, **kwargs):
            time.sleep(0.1)
            return MOCK_NOVELTY
        
        mock_feas_llm.side_effect = slow_feasibility
        mock_impact_llm.side_effect = slow_impact
        mock_cost_llm.side_effect = slow_cost
        mock_novelty_llm.side_effect = slow_novelty
        
        start = time.perf_counter()
        state = run_evaluation(
            proposal_id="perf_test",
            proposal_text="Propuesta para test de rendimiento",
            raw_proposal_data={
                "descripcion_tecnica": "Tech details",
                "recursos_necesarios": "Resources",
                "problema_que_resuelve": "Problem",
                "beneficio_esperado": "Benefit",
                "inversion_inicial_estimada": "100000",
                "descripcion_innovacion": "Innovation",
                "diferenciadores_clave": "Differentiators",
            },
            use_test_graph=True,
        )
        elapsed = time.perf_counter() - start
        
        # Verificar que completó correctamente
        assert state.feasibility_result is not None
        assert state.impact_result is not None
        assert state.cost_result is not None
        assert state.novelty_result is not None
        assert state.weighted_score is not None
        
        # Verificar paralelismo: tiempo debe ser cercano a 0.1s (el agente más lento)
        # NO a 0.4s (suma de los 4 en secuencial)
        # Usamos un umbral generoso: < 0.35s permite paralelismo real
        # Si fuera secuencial, tardaría ~0.4s+
        assert elapsed < 0.35, f"Tiempo {elapsed:.3f}s sugiere ejecución secuencial, no paralela"
        
        # Log para debug
        print(f"\n⏱  Tiempo total fan-out paralelo: {elapsed:.3f}s (esperado ~0.1s, límite 0.35s)")