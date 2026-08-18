import pytest
import os
import tempfile
from proposal_evaluator.config.db_init import init_db, get_db_path
from proposal_evaluator.config.config_repository import (
    get_rubric_by_criterio,
    get_all_rubrics,
    get_active_risk_rules,
    get_risk_rules_by_criterio,
    get_criteria_weights,
    log_audit_event,
    get_audit_events,
)
from proposal_evaluator.schemas import Criterio


@pytest.fixture
def temp_db():
    """Crea una BD temporal para tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    # Set env var for the test
    old_db_path = os.environ.get("DATABASE_PATH")
    os.environ["DATABASE_PATH"] = db_path
    yield db_path
    # Cleanup
    if old_db_path is not None:
        os.environ["DATABASE_PATH"] = old_db_path
    else:
        os.environ.pop("DATABASE_PATH", None)
    if os.path.exists(db_path):
        os.unlink(db_path)


def test_db_init_creates_tables(temp_db):
    """Verifica que init_db crea todas las tablas."""
    conn = init_db()
    cursor = conn.cursor()

    # Verificar tablas existen
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert "rubrics" in tables
    assert "risk_rules" in tables
    assert "criteria_weights" in tables
    assert "audit_events" in tables

    conn.close()


def test_seed_rubrics_inserted(temp_db):
    """Verifica que se insertan 4 rúbricas semilla."""
    init_db()
    rubrics = get_all_rubrics()
    assert len(rubrics) == 4

    criterios = {r.criterio for r in rubrics}
    assert criterios == {Criterio.FEASIBILITY, Criterio.IMPACT, Criterio.COST, Criterio.NOVELTY}


def test_get_rubric_by_criterio(temp_db):
    """Verifica obtención de rúbrica por criterio."""
    init_db()

    rubric = get_rubric_by_criterio(Criterio.FEASIBILITY)
    assert rubric is not None
    assert rubric.criterio == Criterio.FEASIBILITY
    assert "descripcion_tecnica" in rubric.campos_requeridos
    assert len(rubric.escala_guia) == 4

    # Criterio inexistente
    assert get_rubric_by_criterio(Criterio.IMPACT) is not None


def test_seed_risk_rules_inserted(temp_db):
    """Verifica que se insertan reglas de riesgo semilla (3 reglas ilustrativas)."""
    init_db()
    rules = get_active_risk_rules()
    assert len(rules) == 3  # 3 reglas ilustrativas: cost_high, feasibility_low, novelty_patent

    # Verificar regla de costo alto
    cost_rules = get_risk_rules_by_criterio(Criterio.COST)
    high_cost_rule = next((r for r in cost_rules if r.campo_a_evaluar == "costo_estimado_usd"), None)
    assert high_cost_rule is not None
    assert high_cost_rule.operador.value == "gt"
    assert high_cost_rule.valor_umbral == 5000000.0


def test_seed_criteria_weights(temp_db):
    """Verifica pesos por defecto (25% cada uno)."""
    init_db()
    weights = get_criteria_weights()
    assert weights.feasibility == 0.25
    assert weights.impact == 0.25
    assert weights.cost == 0.25
    assert weights.novelty == 0.25


def test_audit_logging(temp_db):
    """Verifica logging de eventos de auditoría."""
    init_db()
    proposal_id = "test_prop_123"

    # Log algunos eventos
    log_audit_event(proposal_id, "agent_evaluation", {"agent": "feasibility", "score": 80})
    log_audit_event(proposal_id, "hitl_triggered", {"reason": "high_cost", "threshold": 5000000})
    log_audit_event(proposal_id, "score_calculated", {"weighted_score": 72.5})

    # Recuperar y verificar
    events = get_audit_events(proposal_id)
    assert len(events) == 3
    assert events[0]["event_type"] == "agent_evaluation"
    assert events[1]["event_type"] == "hitl_triggered"
    assert events[2]["event_type"] == "score_calculated"
    assert events[0]["event_data"]["score"] == 80
    assert events[1]["event_data"]["reason"] == "high_cost"


def test_audit_events_isolated_by_proposal(temp_db):
    """Verifica que eventos se filtran por proposal_id."""
    init_db()
    log_audit_event("prop_a", "event", {"data": 1})
    log_audit_event("prop_b", "event", {"data": 2})

    events_a = get_audit_events("prop_a")
    events_b = get_audit_events("prop_b")

    assert len(events_a) == 1
    assert len(events_b) == 1
    assert events_a[0]["proposal_id"] == "prop_a"
    assert events_b[0]["proposal_id"] == "prop_b"


def test_idempotent_init(temp_db):
    """Verifica que init_db es idempotente (no duplica seed data)."""
    init_db()
    init_db()  # Segunda llamada

    rubrics = get_all_rubrics()
    assert len(rubrics) == 4  # No duplicados

    rules = get_active_risk_rules()
    assert len(rules) == 3  # No duplicados (3 reglas ilustrativas)

    weights = get_criteria_weights()
    assert weights.feasibility == 0.25