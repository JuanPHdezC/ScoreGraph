import os
import structlog
import sqlite3
from contextlib import contextmanager
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

from proposal_evaluator.config.db_init import get_db_path
from proposal_evaluator.config.config_repository import log_audit_event
from proposal_evaluator.schemas import EvaluationState
from proposal_evaluator.agents.feasibility_agent import feasibility_agent
from proposal_evaluator.agents.impact_agent import impact_agent
from proposal_evaluator.agents.cost_agent import cost_agent
from proposal_evaluator.agents.novelty_agent import novelty_agent
from proposal_evaluator.agents.aggregator import calculate_weighted_score
from proposal_evaluator.agents.risk_gate import evaluate_risk_rules
from proposal_evaluator.agents.missing_info_node import (
    missing_info_node,
    should_request_missing_info,
)

logger = structlog.get_logger(__name__)

# ─── Checkpointer management ───
_checkpointer_instance = None
_checkpointer_conn = None


def get_checkpointer(use_memory: bool = False):
    """
    Retorna un checkpointer. Si use_memory=True, usa InMemorySaver (para tests).
    Sino, usa SqliteSaver con conexión persistente.
    """
    global _checkpointer_instance, _checkpointer_conn

    if use_memory:
        return InMemorySaver()

    if _checkpointer_instance is None:
        db_path = get_db_path()
        # Mantener conexión abierta para el checkpointer
        _checkpointer_conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
        )
        _checkpointer_instance = SqliteSaver(_checkpointer_conn)

    return _checkpointer_instance


def close_checkpointer():
    """Cierra la conexión del checkpointer (para cleanup en tests)."""
    global _checkpointer_instance, _checkpointer_conn
    if _checkpointer_conn:
        _checkpointer_conn.close()
        _checkpointer_conn = None
    _checkpointer_instance = None


# ─── Nodos del grafo ───
def init_state_node(state: EvaluationState) -> EvaluationState:
    """Nodo de inicialización: loggea inicio de evaluación."""
    log_audit_event(
        proposal_id=state.proposal_id,
        event_type="evaluation_started",
        event_data={
            "proposal_text_length": len(state.proposal_text),
            "has_raw_data": bool(state.raw_proposal_data),
        },
    )
    logger.info("graph_start", proposal_id=state.proposal_id)
    return state


def synthesizer_placeholder(state: EvaluationState) -> EvaluationState:
    """
    Nodo placeholder del Sintetizador (Fase 4).
    Por ahora solo pasa el estado; en Fase 4 generará el reporte narrativo.
    """
    proposal_id = state.proposal_id
    logger.info("synthesizer_placeholder", proposal_id=proposal_id)
    state.final_report = f"[Fase 4] Reporte pendiente para propuesta {proposal_id}. Score: {state.weighted_score}"
    return state


def hitl_pause_node(state: EvaluationState) -> EvaluationState:
    """
    Nodo de pausa HITL por reglas de riesgo.
    Usa interrupt para esperar aprobación humana.
    """
    proposal_id = state.proposal_id
    payload = {
        "proposal_id": proposal_id,
        "message": "Evaluación requiere validación humana por riesgo de negocio.",
        "hitl_reason": state.hitl_reason,
        "weighted_score": state.weighted_score,
        "individual_scores": {
            "feasibility": state.feasibility_result.score if state.feasibility_result else None,
            "impact": state.impact_result.score if state.impact_result else None,
            "cost": state.cost_result.score if state.cost_result else None,
            "novelty": state.novelty_result.score if state.novelty_result else None,
        },
    }
    log_audit_event(
        proposal_id=proposal_id,
        event_type="hitl_paused",
        event_data=payload,
    )
    logger.warning("hitl_pause", proposal_id=proposal_id, reason=state.hitl_reason)

    user_decision = interrupt(payload)

    if user_decision and isinstance(user_decision, dict):
        approved = user_decision.get("approved", False)
        comment = user_decision.get("comment", "")
        if approved:
            state.hitl_required = False
            state.hitl_reason = None
            logger.info("hitl_approved", proposal_id=proposal_id, comment=comment)
        else:
            logger.warning("hitl_rejected", proposal_id=proposal_id, comment=comment)
    return state


# ─── Funciones de decisión condicional ───
def route_after_agent(state: EvaluationState, agent_name: str) -> Literal["request_info", "continue"]:
    missing = state.missing_fields_by_agent.get(agent_name, [])
    return "request_info" if missing else "continue"


def route_after_aggregator(state: EvaluationState) -> Literal["hitl_pause", "synthesize"]:
    return "hitl_pause" if state.hitl_required else "synthesize"


def route_after_missing_info(state: EvaluationState) -> str:
    for agent in ["feasibility", "impact", "cost", "novelty"]:
        if state.missing_fields_by_agent.get(agent):
            return agent
    return "join_agents"


# ─── Wrappers para agentes: devuelven solo campos modificados (partial state) ───
# Esto permite que LangGraph haga merge automático sin conflictos de escritura concurrente
def _wrap_feasibility_agent(state: EvaluationState) -> dict:
    result = feasibility_agent(state)
    return {
        "feasibility_result": result.feasibility_result,
        "missing_fields_by_agent": result.missing_fields_by_agent,
    }


def _wrap_impact_agent(state: EvaluationState) -> dict:
    result = impact_agent(state)
    return {
        "impact_result": result.impact_result,
        "missing_fields_by_agent": result.missing_fields_by_agent,
    }


def _wrap_cost_agent(state: EvaluationState) -> dict:
    result = cost_agent(state)
    return {
        "cost_result": result.cost_result,
        "missing_fields_by_agent": result.missing_fields_by_agent,
    }


def _wrap_novelty_agent(state: EvaluationState) -> dict:
    result = novelty_agent(state)
    return {
        "novelty_result": result.novelty_result,
        "missing_fields_by_agent": result.missing_fields_by_agent,
    }


# ─── Reducer para merge de dicts (missing_fields_by_agent) ───
def _merge_dicts(left: dict, right: dict) -> dict:
    """Merge dicts - cada agente escribe en su clave distinta."""
    return {**left, **right}


# ─── Construcción del StateGraph ───
def _build_graph(use_memory: bool = False) -> StateGraph:
    """
    Construye y compila el grafo de evaluación con FAN-OUT PARALELO REAL.
    
    Topología:
    - init → [feasibility, impact, cost, novelty] (paralelo, 4 ramas)
    - Cada agente → missing_info (si faltan campos) O join_agents
    - join_agents (barrera natural: espera a las 4 ramas) → aggregator → risk_gate → ...
    
    Los agentes devuelven partial state updates (solo campos que modifican).
    LangGraph hace merge automático de partial state updates.
    El campo `missing_fields_by_agent` usa BinaryOperatorAggregate con merge de dicts.
    """
    import operator
    from langgraph.channels.binop import BinaryOperatorAggregate
    
    graph = StateGraph(EvaluationState)
    
    # Configurar reducer personalizado para missing_fields_by_agent usando BinaryOperatorAggregate
    # operator.or_ hace merge de dicts (Python 3.9+): dict1 | dict2
    graph.channels["missing_fields_by_agent"] = BinaryOperatorAggregate(dict, operator.or_)

    # Nodos (usar wrapped agents para partial state updates)
    graph.add_node("init", init_state_node)
    graph.add_node("feasibility", _wrap_feasibility_agent)
    graph.add_node("impact", _wrap_impact_agent)
    graph.add_node("cost", _wrap_cost_agent)
    graph.add_node("novelty", _wrap_novelty_agent)
    graph.add_node("missing_info", missing_info_node)
    graph.add_node("aggregator", calculate_weighted_score)
    graph.add_node("risk_gate", evaluate_risk_rules)
    graph.add_node("hitl_pause", hitl_pause_node)
    graph.add_node("synthesize", synthesizer_placeholder)
    graph.add_node("join_agents", lambda s: s)  # Nodo barrera (fan-in)

    # Edge desde START
    graph.add_edge(START, "init")

    # ─── FAN-OUT PARALELO REAL: init → 4 agentes simultáneos ───
    graph.add_edge("init", "feasibility")
    graph.add_edge("init", "impact")
    graph.add_edge("init", "cost")
    graph.add_edge("init", "novelty")

    # ─── Cada agente decide: missing_info O join_agents ───
    for agent in ["feasibility", "impact", "cost", "novelty"]:
        graph.add_conditional_edges(
            agent,
            lambda s, a=agent: route_after_agent(s, a),
            {
                "request_info": "missing_info",
                "continue": "join_agents",
            },
        )

    # ─── missing_info re-ejecuta solo el agente que faltaba ───
    graph.add_conditional_edges(
        "missing_info",
        route_after_missing_info,
        {
            "feasibility": "feasibility",
            "impact": "impact",
            "cost": "cost",
            "novelty": "novelty",
            "join_agents": "join_agents",
        },
    )

    # ─── FAN-IN NATIVO: join_agents es barrera implícita ───
    # LangGraph espera a que TODAS las 4 ramas lleguen a join_agents
    graph.add_edge("join_agents", "aggregator")

    # Aggregator → risk_gate
    graph.add_edge("aggregator", "risk_gate")

    # Risk gate → HITL pause o synthesize
    graph.add_conditional_edges(
        "risk_gate",
        route_after_aggregator,
        {
            "hitl_pause": "hitl_pause",
            "synthesize": "synthesize",
        },
    )

    # HITL pause → synthesize (tras aprobación)
    graph.add_edge("hitl_pause", "synthesize")

    # Synthesize → END
    graph.add_edge("synthesize", END)

    # Compilar con checkpointer
    checkpointer = get_checkpointer(use_memory=use_memory)
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled


def build_graph() -> StateGraph:
    """Grafo de producción con SQLite checkpointer."""
    return _build_graph(use_memory=False)


def build_test_graph() -> StateGraph:
    """Grafo para tests con InMemorySaver."""
    return _build_graph(use_memory=True)


# ─── Funciones públicas de API ───
_graph_instance = None
_test_graph_instance = None


def get_graph():
    """Singleton del grafo compilado (producción con SQLite)."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_graph()
    return _graph_instance


def get_test_graph():
    """Singleton del grafo compilado (tests con InMemorySaver)."""
    global _test_graph_instance
    if _test_graph_instance is None:
        _test_graph_instance = build_test_graph()
    return _test_graph_instance


def run_evaluation(
    proposal_id: str,
    proposal_text: str,
    raw_proposal_data: dict | None = None,
    use_test_graph: bool = False,
) -> EvaluationState:
    """
    Inicia una nueva evaluación.
    Retorna el estado final (o pausado en interrupt).
    """
    initial_state = EvaluationState(
        proposal_id=proposal_id,
        proposal_text=proposal_text,
        raw_proposal_data=raw_proposal_data or {},
    )

    graph = get_test_graph() if use_test_graph else get_graph()
    config = {"configurable": {"thread_id": proposal_id}}

    result = graph.invoke(initial_state, config=config)

    if isinstance(result, dict):
        return EvaluationState(**result)
    return result


def resume_evaluation(
    proposal_id: str,
    new_data: dict,
    use_test_graph: bool = False,
) -> EvaluationState:
    """
    Reanuda una evaluación pausada con nueva información del usuario.
    new_data debe contener raw_proposal_data y opcionalmente decision de HITL.
    """
    graph = get_test_graph() if use_test_graph else get_graph()
    config = {"configurable": {"thread_id": proposal_id}}

    result = graph.invoke(new_data, config=config)

    if isinstance(result, dict):
        return EvaluationState(**result)
    return result


def get_evaluation_state(proposal_id: str, use_test_graph: bool = False) -> EvaluationState | None:
    graph = get_test_graph() if use_test_graph else get_graph()
    config = {"configurable": {"thread_id": proposal_id}}

    try:
        state_snapshot = graph.get_state(config)
        if state_snapshot and state_snapshot.values:
            vals = state_snapshot.values
            if isinstance(vals, dict):
                return EvaluationState(**vals)
            return vals
    except Exception:
        pass
    return None