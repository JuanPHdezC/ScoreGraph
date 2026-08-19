import sqlite3
from typing import Optional
from pathlib import Path
from contextlib import contextmanager

from proposal_evaluator.config.db_init import get_db_path

# ─── Conexión a BD ───
@contextmanager
def get_db():
    """Context manager para conexión a BD."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ─── Métricas ───
def get_metrics_summary() -> dict:
    """
    Calcula métricas agregadas sobre audit_events.
    
    Returns:
        dict con métricas de resumen
    """
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Total evaluaciones completadas (tienen score_calculated)
        cursor.execute("""
            SELECT COUNT(DISTINCT proposal_id) 
            FROM audit_events 
            WHERE event_type = 'score_calculated'
        """)
        total_completed = cursor.fetchone()[0] or 0
        
        # Evaluaciones que dispararon HITL
        cursor.execute("""
            SELECT COUNT(DISTINCT proposal_id) 
            FROM audit_events 
            WHERE event_type = 'hitl_triggered'
        """)
        total_hitl = cursor.fetchone()[0] or 0
        
        # Evaluaciones que tuvieron reintentos por info faltante
        cursor.execute("""
            SELECT COUNT(DISTINCT proposal_id) 
            FROM audit_events 
            WHERE event_type = 'missing_info_requested'
        """)
        total_retries = cursor.fetchone()[0] or 0
        
        # Tasa HITL
        hitl_rate = (total_hitl / total_completed * 100) if total_completed > 0 else 0.0
        
        # Tasa de reintentos
        retry_rate = (total_retries / total_completed * 100) if total_completed > 0 else 0.0
        
        # Distribución de scores promedio por criterio (solo completadas)
        criteria_scores = {}
        for criterion in ["feasibility", "impact", "cost", "novelty"]:
            cursor.execute(f"""
                SELECT AVG(CAST(json_extract(event_data, '$.individual_scores.{criterion}') AS REAL))
                FROM audit_events
                WHERE event_type = 'score_calculated'
                AND json_extract(event_data, '$.individual_scores.{criterion}') IS NOT NULL
            """)
            avg_score = cursor.fetchone()[0]
            criteria_scores[criterion] = round(avg_score, 2) if avg_score is not None else None
        
        # Score ponderado promedio
        cursor.execute("""
            SELECT AVG(CAST(json_extract(event_data, '$.weighted_score') AS REAL))
            FROM audit_events
            WHERE event_type = 'score_calculated'
        """)
        avg_weighted = cursor.fetchone()[0]
        
        return {
            "total_evaluations_completed": total_completed,
            "hitl_triggered_count": total_hitl,
            "hitl_rate_percent": round(hitl_rate, 2),
            "missing_info_retries_count": total_retries,
            "retry_rate_percent": round(retry_rate, 2),
            "average_scores_by_criterion": criteria_scores,
            "average_weighted_score": round(avg_weighted, 2) if avg_weighted is not None else None,
        }


def get_evaluation_events(proposal_id: str) -> list[dict]:
    """Obtiene todos los eventos de auditoría para una propuesta."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM audit_events 
            WHERE proposal_id = ? 
            ORDER BY created_at
        """, (proposal_id,))
        rows = cursor.fetchall()
        return [
            {
                "id": row["id"],
                "proposal_id": row["proposal_id"],
                "event_type": row["event_type"],
                "event_data": row["event_data"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]


def get_all_proposals() -> list[dict]:
    """Lista todas las propuestas con su estado final."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT proposal_id,
                   MAX(CASE WHEN event_type = 'score_calculated' THEN json_extract(event_data, '$.weighted_score') END) as weighted_score,
                   MAX(CASE WHEN event_type = 'hitl_triggered' THEN 1 ELSE 0 END) as hitl_triggered,
                   MAX(CASE WHEN event_type = 'evaluation_started' THEN created_at END) as started_at
            FROM audit_events
            GROUP BY proposal_id
            ORDER BY started_at DESC
        """)
        rows = cursor.fetchall()
        return [
            {
                "proposal_id": row["proposal_id"],
                "weighted_score": row["weighted_score"],
                "hitl_triggered": bool(row["hitl_triggered"]),
                "started_at": row["started_at"],
            }
            for row in rows
        ]