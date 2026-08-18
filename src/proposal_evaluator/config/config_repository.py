import json
import sqlite3
import os
from pathlib import Path
from typing import Optional
from proposal_evaluator.schemas import (
    Rubric,
    RiskRule,
    CriteriaWeights,
    Criterio,
    OperadorRiesgo,
)
from proposal_evaluator.config.db_init import get_db_path


def _get_conn() -> sqlite3.Connection:
    """Obtiene una conexión a la BD con row_factory."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def get_rubric_by_criterio(criterio: Criterio) -> Optional[Rubric]:
    """Obtiene la rúbrica activa para un criterio dado."""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rubrics WHERE criterio = ?", (criterio.value,))
        row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_rubric(row)
    finally:
        conn.close()


def get_all_rubrics() -> list[Rubric]:
    """Obtiene todas las rúbricas."""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rubrics")
        rows = cursor.fetchall()
        return [_row_to_rubric(row) for row in rows]
    finally:
        conn.close()


def _row_to_rubric(row: sqlite3.Row) -> Rubric:
    return Rubric(
        id=row["id"],
        criterio=Criterio(row["criterio"]),
        campos_requeridos=json.loads(row["campos_requeridos"]),
        campos_opcionales=json.loads(row["campos_opcionales"]),
        criterios_de_evaluacion=row["criterios_de_evaluacion"],
        escala_guia=json.loads(row["escala_guia"]),
    )


def get_active_risk_rules() -> list[RiskRule]:
    """Obtiene todas las reglas de riesgo activas."""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM risk_rules WHERE activa = 1")
        rows = cursor.fetchall()
        return [_row_to_risk_rule(row) for row in rows]
    finally:
        conn.close()


def get_risk_rules_by_criterio(criterio: Criterio) -> list[RiskRule]:
    """Obtiene reglas de riesgo activas para un criterio específico."""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM risk_rules WHERE criterio = ? AND activa = 1", (criterio.value,))
        rows = cursor.fetchall()
        return [_row_to_risk_rule(row) for row in rows]
    finally:
        conn.close()


def _row_to_risk_rule(row: sqlite3.Row) -> RiskRule:
    return RiskRule(
        id=row["id"],
        criterio=Criterio(row["criterio"]),
        campo_a_evaluar=row["campo_a_evaluar"],
        operador=OperadorRiesgo(row["operador"]),
        valor_umbral=json.loads(row["valor_umbral"]),
        descripcion_razon=row["descripcion_razon"],
    )


def get_criteria_weights() -> CriteriaWeights:
    """Obtiene los pesos actuales de agregación."""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM criteria_weights WHERE id = 'default'")
        row = cursor.fetchone()
        if row is None:
            # Fallback a pesos por defecto
            return CriteriaWeights()
        return CriteriaWeights(
            feasibility=row["feasibility"],
            impact=row["impact"],
            cost=row["cost"],
            novelty=row["novelty"],
        )
    finally:
        conn.close()


def log_audit_event(proposal_id: str, event_type: str, event_data: dict) -> None:
    """Registra un evento de auditoría en la BD."""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_events (proposal_id, event_type, event_data)
            VALUES (?, ?, ?)
        """, (proposal_id, event_type, json.dumps(event_data)))
        conn.commit()
    finally:
        conn.close()


def get_audit_events(proposal_id: str) -> list[dict]:
    """Obtiene todos los eventos de auditoría para una propuesta."""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM audit_events WHERE proposal_id = ? ORDER BY created_at
        """, (proposal_id,))
        rows = cursor.fetchall()
        return [
            {
                "id": row["id"],
                "proposal_id": row["proposal_id"],
                "event_type": row["event_type"],
                "event_data": json.loads(row["event_data"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    finally:
        conn.close()