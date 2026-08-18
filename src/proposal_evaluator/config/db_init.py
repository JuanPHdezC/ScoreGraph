import sqlite3
import os
from pathlib import Path
from proposal_evaluator.schemas import (
    Rubric,
    RiskRule,
    CriteriaWeights,
    Criterio,
    OperadorRiesgo,
)


def get_db_path() -> str:
    """Obtiene la ruta de la BD desde variable de entorno o usa default."""
    return os.getenv("DATABASE_PATH", "./data/proposal_evaluator.db")


def init_db(db_path: str | None = None) -> sqlite3.Connection:
    """
    Inicializa la base de datos SQLite creando todas las tablas necesarias
    e insertando datos semilla (seed data).
    """
    if db_path is None:
        db_path = get_db_path()

    # Asegurar que el directorio existe
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Tabla: rubrics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rubrics (
            id TEXT PRIMARY KEY,
            criterio TEXT NOT NULL,
            campos_requeridos TEXT NOT NULL,  -- JSON array
            campos_opcionales TEXT NOT NULL,  -- JSON array
            criterios_de_evaluacion TEXT NOT NULL,
            escala_guia TEXT NOT NULL,        -- JSON object
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Tabla: risk_rules
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS risk_rules (
            id TEXT PRIMARY KEY,
            criterio TEXT NOT NULL,
            campo_a_evaluar TEXT NOT NULL,
            operador TEXT NOT NULL,
            valor_umbral TEXT NOT NULL,       -- JSON value (float, bool, string)
            descripcion_razon TEXT NOT NULL,
            activa INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Tabla: criteria_weights
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS criteria_weights (
            id TEXT PRIMARY KEY DEFAULT 'default',
            feasibility REAL NOT NULL,
            impact REAL NOT NULL,
            cost REAL NOT NULL,
            novelty REAL NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Tabla: audit_events
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            proposal_id TEXT NOT NULL,
            event_type TEXT NOT NULL,         -- ej: 'agent_evaluation', 'hitl_triggered', 'score_calculated', 'field_missing'
            event_data TEXT NOT NULL,         -- JSON con detalles
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Índices para audit_events
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_proposal ON audit_events(proposal_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_events(event_type)")

    conn.commit()

    # Insertar seed data si las tablas están vacías
    _seed_rubrics(cursor)
    _seed_risk_rules(cursor)
    _seed_criteria_weights(cursor)

    conn.commit()
    return conn


def _seed_rubrics(cursor: sqlite3.Cursor) -> None:
    """Inserta rúbricas por defecto para los 4 criterios."""
    cursor.execute("SELECT COUNT(*) FROM rubrics")
    if cursor.fetchone()[0] > 0:
        return

    import json

    rubrics = [
        Rubric(
            id="rubric_feasibility_v1",
            criterio=Criterio.FEASIBILITY,
            campos_requeridos=["descripcion_tecnica", "recursos_necesarios"],
            campos_opcionales=["equipo_disponible", "cronograma_estimado", "riesgos_tecnicos"],
            criterios_de_evaluacion=(
                "Evalúa la factibilidad técnica y operativa de la propuesta. Considera: "
                "1) Madurez de la tecnología propuesta, 2) Disponibilidad de recursos y talento, "
                "3) Complejidad de implementación, 4) Riesgos técnicos identificados, "
                "5) Cronograma realista. Un score alto indica que la propuesta es técnicamente "
                "viable con recursos disponibles o accesibles."
            ),
            escala_guia={
                "0-30": "Muy baja factibilidad: tecnología inmadura, recursos no disponibles, riesgos críticos sin mitigación",
                "31-60": "Factibilidad moderada: algunos desafíos técnicos significativos pero abordables con inversión",
                "61-85": "Alta factibilidad: tecnología probada, recursos disponibles, plan de implementación claro",
                "86-100": "Factibilidad muy alta: implementación directa con capacidades actuales, bajo riesgo técnico"
            }
        ),
        Rubric(
            id="rubric_impact_v1",
            criterio=Criterio.IMPACT,
            campos_requeridos=["problema_que_resuelve", "beneficio_esperado"],
            campos_opcionales=["tamano_mercado", "usuarios_afectados", "ventaja_competitiva", "kpis_objetivo"],
            criterios_de_evaluacion=(
                "Evalúa el impacto potencial de la propuesta en el negocio y los usuarios. Considera: "
                "1) Magnitud del problema resuelto, 2) Tamaño de la oportunidad/mercado, "
                "3) Número de usuarios/beneficiarios afectados, 4) Ventaja competitiva sostenible, "
                "5) KPIs medibles de éxito. Un score alto indica impacto transformador y medible."
            ),
            escala_guia={
                "0-30": "Impacto bajo: problema menor, mercado pequeño, beneficio incremental sin diferenciación",
                "31-60": "Impacto moderado: problema relevante, mercado mediano, mejora medible pero no transformadora",
                "61-85": "Alto impacto: problema significativo, mercado grande, ventaja competitiva clara",
                "86-100": "Impacto transformador: resuelve problema crítico, mercado masivo, ventaja única y defendible"
            }
        ),
        Rubric(
            id="rubric_cost_v1",
            criterio=Criterio.COST,
            campos_requeridos=["inversion_inicial_estimada"],
            campos_opcionales=["costo_operativo_anual", "roi_estimado_meses", "fuentes_financiacion", "desglose_costos"],
            criterios_de_evaluacion=(
                "Evalúa el costo total de propiedad y viabilidad económica. Considera: "
                "1) Inversión inicial (CAPEX), 2) Costos operativos recurrentes (OPEX), "
                "3) Retorno de inversión (ROI) y tiempo de recupero, 4) Estructura de costos, "
                "5) Disponibilidad de financiación. IMPORTANTE: Si la propuesta incluye "
                "costo_estimado_usd, úsalo directamente. Si no, estima basado en la información disponible. "
                "Un score alto indica costo bajo/razonable para el valor esperado."
            ),
            escala_guia={
                "0-30": "Costo muy alto: inversión prohibitiva, ROI > 36 meses, estructura de costos insostenible",
                "31-60": "Costo elevado: inversión significativa, ROI 18-36 meses, requiere financiación externa",
                "61-85": "Costo razonable: inversión moderada, ROI 12-18 meses, financiación interna viable",
                "86-100": "Costo óptimo: inversión baja, ROI < 12 meses, autosostenible desde el inicio"
            }
        ),
        Rubric(
            id="rubric_novelty_v1",
            criterio=Criterio.NOVELTY,
            campos_requeridos=["descripcion_innovacion", "diferenciadores_clave"],
            campos_opcionales=["patentes_relacionadas", "estado_del_arte", "busqueda_previa_realizada"],
            criterios_de_evaluacion=(
                "Evalúa el grado de novedad e innovación de la propuesta. Considera: "
                "1) Grado de diferenciación frente al estado del arte actual, "
                "2) Existencia de patentes o soluciones similares en el mercado, "
                "3) Potencial de propiedad intelectual generada, "
                "4) Primera en el mercado (first-mover advantage). "
                "IMPORTANTE: Indica en colision_patente_detectada si identificas riesgo de infracción. "
                "Lista en fuentes_consultadas las bases de datos/repositorios consultados. "
                "Un score alto indica innovación radical y defendible."
            ),
            escala_guia={
                "0-30": "Baja novedad: mejora incremental, soluciones equivalentes existentes, sin IP defendible",
                "31-60": "Novedad moderada: combinación novedosa de elementos existentes, alguna diferenciación",
                "61-85": "Alta novedad: enfoque significativamente nuevo, pocas soluciones comparables, IP potencial",
                "86-100": "Novedad radical: paradigma completamente nuevo, first-in-class, IP fuerte y defendible"
            }
        ),
    ]

    for rubric in rubrics:
        cursor.execute("""
            INSERT INTO rubrics (id, criterio, campos_requeridos, campos_opcionales, criterios_de_evaluacion, escala_guia)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            rubric.id,
            rubric.criterio.value,
            json.dumps(rubric.campos_requeridos),
            json.dumps(rubric.campos_opcionales),
            rubric.criterios_de_evaluacion,
            json.dumps(rubric.escala_guia),
        ))


def _seed_risk_rules(cursor: sqlite3.Cursor) -> None:
    """Inserta reglas de riesgo de negocio por defecto."""
    cursor.execute("SELECT COUNT(*) FROM risk_rules")
    if cursor.fetchone()[0] > 0:
        return

    import json

    risk_rules = [
        RiskRule(
            id="risk_cost_high",
            criterio=Criterio.COST,
            campo_a_evaluar="costo_estimado_usd",
            operador=OperadorRiesgo.GT,
            valor_umbral=5000000.0,
            descripcion_razon="Costo estimado superior a $5M requiere validación ejecutiva"
        ),
        RiskRule(
            id="risk_feasibility_low",
            criterio=Criterio.FEASIBILITY,
            campo_a_evaluar="score",
            operador=OperadorRiesgo.LT,
            valor_umbral=30,
            descripcion_razon="Factibilidad muy baja (score < 30) requiere revisión técnica"
        ),
        RiskRule(
            id="risk_novelty_patent_collision",
            criterio=Criterio.NOVELTY,
            campo_a_evaluar="colision_patente_detectada",
            operador=OperadorRiesgo.IS_TRUE,
            valor_umbral=True,
            descripcion_razon="Posible colisión de patente detectada, requiere revisión legal"
        ),
        RiskRule(
            id="risk_impact_very_high",
            criterio=Criterio.IMPACT,
            campo_a_evaluar="score",
            operador=OperadorRiesgo.GT,
            valor_umbral=90,
            descripcion_razon="Impacto extremadamente alto (score > 90) requiere validación de expectativas"
        ),
    ]

    for rule in risk_rules:
        cursor.execute("""
            INSERT INTO risk_rules (id, criterio, campo_a_evaluar, operador, valor_umbral, descripcion_razon, activa)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        """, (
            rule.id,
            rule.criterio.value,
            rule.campo_a_evaluar,
            rule.operador.value,
            json.dumps(rule.valor_umbral),
            rule.descripcion_razon,
        ))


def _seed_criteria_weights(cursor: sqlite3.Cursor) -> None:
    """Inserta pesos por defecto (25% cada uno)."""
    cursor.execute("SELECT COUNT(*) FROM criteria_weights")
    if cursor.fetchone()[0] > 0:
        return

    cursor.execute("""
        INSERT INTO criteria_weights (id, feasibility, impact, cost, novelty)
        VALUES ('default', 0.25, 0.25, 0.25, 0.25)
    """)


if __name__ == "__main__":
    conn = init_db()
    print(f"Base de datos inicializada correctamente en: {get_db_path()}")
    conn.close()