import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from proposal_evaluator.schemas import EvaluationState, FeasibilityEvaluation, ImpactEvaluation, CostEvaluation, NoveltyEvaluation
from proposal_evaluator.api.main import app
from proposal_evaluator.observability.metrics import get_metrics_summary


class TestAPIEndpoints:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    @patch("proposal_evaluator.api.main.run_evaluation")
    def test_create_evaluation_success(self, mock_run, client):
        from proposal_evaluator.schemas import EvaluationState, FeasibilityEvaluation, ImpactEvaluation, CostEvaluation, NoveltyEvaluation
        
        mock_state = EvaluationState(
            proposal_id="test_api_1",
            proposal_text="Test",
            raw_proposal_data={},
            feasibility_result=FeasibilityEvaluation(score=80, summary="OK", confidence_fields_present=True),
            impact_result=ImpactEvaluation(score=85, summary="OK", confidence_fields_present=True),
            cost_result=CostEvaluation(score=70, summary="OK", confidence_fields_present=True),
            novelty_result=NoveltyEvaluation(score=75, summary="OK", confidence_fields_present=True),
            weighted_score=77.5,
        )
        mock_run.return_value = mock_state
        
        response = client.post("/evaluations", json={
            "proposal_text": "Propuesta de prueba para API",
            "raw_proposal_data": {}
        })
        
        assert response.status_code == 201
        data = response.json()
        # El API genera UUID aleatorio, verificar que sea un UUID válido
        assert "proposal_id" in data
        assert data["status"] == "completed"
        assert data["state"]["weighted_score"] == 77.5

    @patch("proposal_evaluator.api.main.get_evaluation_state")
    def test_get_evaluation_not_found(self, mock_get, client):
        mock_get.return_value = None
        
        response = client.get("/evaluations/nonexistent")
        
        assert response.status_code == 404

    @patch("proposal_evaluator.api.main.get_evaluation_state")
    def test_get_evaluation_found(self, mock_get, client):
        from proposal_evaluator.schemas import EvaluationState, FeasibilityEvaluation, ImpactEvaluation, CostEvaluation, NoveltyEvaluation
        
        mock_state = EvaluationState(
            proposal_id="test_get_1",
            proposal_text="Test",
            raw_proposal_data={},
            feasibility_result=FeasibilityEvaluation(score=80, summary="OK", confidence_fields_present=True),
            impact_result=ImpactEvaluation(score=85, summary="OK", confidence_fields_present=True),
            cost_result=CostEvaluation(score=70, summary="OK", confidence_fields_present=True),
            novelty_result=NoveltyEvaluation(score=75, summary="OK", confidence_fields_present=True),
            weighted_score=77.5,
        )
        mock_get.return_value = mock_state
        
        response = client.get("/evaluations/test_get_1")
        
        assert response.status_code == 200
        data = response.json()
        assert data["proposal_id"] == "test_get_1"
        assert data["status"] == "completed"

    @patch("proposal_evaluator.api.main.resume_evaluation")
    @patch("proposal_evaluator.api.main.get_evaluation_state")
    def test_provide_info_success(self, mock_get, mock_resume, client):
        from proposal_evaluator.schemas import EvaluationState, FeasibilityEvaluation, ImpactEvaluation, CostEvaluation, NoveltyEvaluation
        
        # Estado inicial: pausado por campos faltantes
        initial_state = EvaluationState(
            proposal_id="test_info_1",
            proposal_text="Test",
            raw_proposal_data={},
            hitl_required=False,
            missing_fields_by_agent={"feasibility": ["descripcion_tecnica"]},
        )
        mock_get.return_value = initial_state
        
        # Estado tras reanudar
        resumed_state = EvaluationState(
            proposal_id="test_info_1",
            proposal_text="Test",
            raw_proposal_data={"descripcion_tecnica": "Added"},
            feasibility_result=FeasibilityEvaluation(score=80, summary="OK", confidence_fields_present=True),
            impact_result=ImpactEvaluation(score=85, summary="OK", confidence_fields_present=True),
            cost_result=CostEvaluation(score=70, summary="OK", confidence_fields_present=True),
            novelty_result=NoveltyEvaluation(score=75, summary="OK", confidence_fields_present=True),
            weighted_score=77.5,
        )
        mock_resume.return_value = resumed_state
        
        response = client.post("/evaluations/test_info_1/provide-info", json={
            "raw_proposal_data": {"descripcion_tecnica": "Added"}
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"

    @patch("proposal_evaluator.api.main.get_evaluation_state")
    def test_hitl_approve_not_found(self, mock_get, client):
        mock_get.return_value = None
        
        response = client.post("/evaluations/nonexistent/hitl-approve", json={"approved": True})
        assert response.status_code == 404

    @patch("proposal_evaluator.api.main.resume_evaluation")
    @patch("proposal_evaluator.api.main.get_evaluation_state")
    def test_hitl_approve_success(self, mock_get, mock_resume, client):
        from proposal_evaluator.schemas import EvaluationState
        
        initial_state = EvaluationState(
            proposal_id="test_hitl_1",
            proposal_text="Test",
            raw_proposal_data={},
            hitl_required=True,
            hitl_reason="Costo alto",
        )
        mock_get.return_value = initial_state
        
        final_state = EvaluationState(
            proposal_id="test_hitl_1",
            proposal_text="Test",
            raw_proposal_data={},
            hitl_required=False,
        )
        mock_resume.return_value = final_state
        
        response = client.post("/evaluations/test_hitl_1/hitl-approve", json={
            "approved": True,
            "comment": "Aprobado"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"

    @patch("proposal_evaluator.api.main.resume_evaluation")
    @patch("proposal_evaluator.api.main.get_evaluation_state")
    def test_hitl_reject(self, mock_get, mock_resume, client):
        from proposal_evaluator.schemas import EvaluationState
        
        initial_state = EvaluationState(
            proposal_id="test_hitl_2",
            proposal_text="Test",
            raw_proposal_data={},
            hitl_required=True,
            hitl_reason="Costo alto",
        )
        mock_get.return_value = initial_state
        
        final_state = EvaluationState(
            proposal_id="test_hitl_2",
            proposal_text="Test",
            raw_proposal_data={},
            hitl_required=True,  # Se mantiene True si se rechaza
        )
        mock_resume.return_value = final_state
        
        response = client.post("/evaluations/test_hitl_2/hitl-approve", json={
            "approved": False,
            "comment": "Rechazado"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "paused_hitl"

    @patch("proposal_evaluator.api.main.get_metrics_summary")
    def test_metrics_endpoint(self, mock_metrics, client):
        mock_metrics.return_value = {
            "total_evaluations_completed": 10,
            "hitl_triggered_count": 2,
            "hitl_rate_percent": 20.0,
            "missing_info_retries_count": 5,
            "retry_rate_percent": 50.0,
            "average_scores_by_criterion": {"feasibility": 75.0, "impact": 80.0, "cost": 70.0, "novelty": 72.0},
            "average_weighted_score": 74.25,
        }
        
        response = client.get("/metrics/summary")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_evaluations_completed"] == 10
        assert data["hitl_rate_percent"] == 20.0

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @patch("proposal_evaluator.api.main.get_evaluation_state")
    def test_hitl_approve_invalid_action_for_narrative_returns_422(self, mock_get, client):
        """Test que action inválida para narrative_hallucination devuelve 422 con acciones válidas."""
        from proposal_evaluator.schemas import EvaluationState
        
        initial_state = EvaluationState(
            proposal_id="test_narrative_invalid",
            proposal_text="Test",
            raw_proposal_data={},
            hitl_required=True,
            hitl_type="narrative_hallucination",
            hitl_reason="Narrativa contiene cifras no verificadas",
        )
        mock_get.return_value = initial_state
        
        response = client.post("/evaluations/test_narrative_invalid/hitl-approve", json={
            "action": "approve",  # Inválido para narrative_hallucination
        })
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data["detail"]
        assert "narrative_hallucination" in data["detail"]["error"]
        assert "approve_text" in data["detail"]["valid_actions"]
        assert "regenerate" in data["detail"]["valid_actions"]

    @patch("proposal_evaluator.api.main.resume_evaluation")
    @patch("proposal_evaluator.api.main.get_evaluation_state")
    def test_legacy_state_without_hitl_type_defaults_to_risk(self, mock_get, mock_resume, client):
        """Test que estado legacy sin hitl_type (None) se trata como 'risk'."""
        from proposal_evaluator.schemas import EvaluationState
        
        initial_state = EvaluationState(
            proposal_id="test_legacy",
            proposal_text="Test",
            raw_proposal_data={},
            hitl_required=True,
            hitl_type=None,  # Legacy: sin hitl_type
            hitl_reason="Costo alto",
        )
        mock_get.return_value = initial_state
        
        final_state = EvaluationState(
            proposal_id="test_legacy",
            proposal_text="Test",
            raw_proposal_data={},
            hitl_required=False,
        )
        mock_resume.return_value = final_state
        
        # Enviar request SIN action (compatibilidad legacy)
        response = client.post("/evaluations/test_legacy/hitl-approve", json={
            "approved": True,
            "comment": "Aprobado legacy"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"

    @patch("proposal_evaluator.api.main.resume_evaluation")
    @patch("proposal_evaluator.api.main.get_evaluation_state")
    def test_api_hitl_approve_dispatches_by_type(self, mock_get, mock_resume, client):
        """Test: API despacha correctamente según hitl_type (risk vs narrative_hallucination)."""
        from proposal_evaluator.schemas import EvaluationState
        
        # Caso 1: hitl_type="risk" → solo permite approve/reject
        risk_state = EvaluationState(
            proposal_id="test_risk",
            proposal_text="Test",
            raw_proposal_data={},
            hitl_required=True,
            hitl_type="risk",
            hitl_reason="Costo alto",
        )
        mock_get.return_value = risk_state
        
        # action="approve_text" en risk debe fallar
        response = client.post("/evaluations/test_risk/hitl-approve", json={"action": "approve_text"})
        assert response.status_code == 422
        data = response.json()
        assert "approve_text" not in data["detail"]["valid_actions"]
        assert "approve" in data["detail"]["valid_actions"]
        assert "reject" in data["detail"]["valid_actions"]
        
        # action="approve" en risk debe funcionar
        with patch("proposal_evaluator.api.main.resume_evaluation") as mock_resume, \
             patch("proposal_evaluator.api.main.get_evaluation_state") as mock_get2:
            final_risk = EvaluationState(proposal_id="test_risk", proposal_text="Test", raw_proposal_data={}, hitl_required=False)
            mock_resume.return_value = final_risk
            mock_get2.return_value = EvaluationState(proposal_id="test_risk", proposal_text="Test", raw_proposal_data={}, hitl_required=True, hitl_type="risk", hitl_reason="Costo alto")
            
            response = client.post("/evaluations/test_risk/hitl-approve", json={"action": "approve"})
            assert response.status_code == 200
        
        # Caso 2: hitl_type="narrative_hallucination" → solo permite approve_text/regenerate
        narrative_state = EvaluationState(
            proposal_id="test_narrative",
            proposal_text="Test",
            raw_proposal_data={},
            hitl_required=True,
            hitl_type="narrative_hallucination",
            hitl_reason="Narrativa con cifras",
        )
        mock_get.return_value = narrative_state
        
        # action="approve" en narrative debe fallar
        response = client.post("/evaluations/test_narrative/hitl-approve", json={"action": "approve"})
        assert response.status_code == 422
        data = response.json()
        assert "approve" not in data["detail"]["valid_actions"]
        assert "approve_text" in data["detail"]["valid_actions"]
        assert "regenerate" in data["detail"]["valid_actions"]
        
        # action="approve_text" en narrative debe funcionar
        with patch("proposal_evaluator.api.main.resume_evaluation") as mock_resume, \
             patch("proposal_evaluator.api.main.get_evaluation_state") as mock_get2:
            final_narr = EvaluationState(proposal_id="test_narrative", proposal_text="Test", raw_proposal_data={}, hitl_required=False)
            mock_resume.return_value = final_narr
            mock_get2.return_value = narrative_state
            
            response = client.post("/evaluations/test_narrative/hitl-approve", json={"action": "approve_text"})
            assert response.status_code == 200


class TestMetrics:
    @patch("proposal_evaluator.observability.metrics.get_db")
    def test_metrics_summary_empty(self, mock_get_db):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        mock_cursor.fetchone.side_effect = [
            (0,),  # total_completed
            (0,),  # total_hitl
            (0,),  # total_retries
            (None,), (None,), (None,), (None,),  # avg scores
            (None,),  # avg weighted
        ]
        
        result = get_metrics_summary()
        
        assert result["total_evaluations_completed"] == 0
        assert result["hitl_rate_percent"] == 0.0
        assert result["retry_rate_percent"] == 0.0