# Proposal Evaluator - Sistema Agéntico de Evaluación de Propuestas de Innovación

MVP de un sistema multi-agente para evaluar propuestas de innovación bajo 4 criterios independientes:
**Factibilidad, Impacto, Costo y Novedad**, produciendo un scoring unificado y reporte narrativo con soporte Human-in-the-Loop (HITL).

## Stack Tecnológico

- **Python 3.11+**
- **LangGraph** - Orquestación del grafo de agentes
- **Pydantic v2** - Validación de esquemas y structured outputs
- **Anthropic Claude API** - Proveedor LLM (modelo económico: claude-3-haiku)
- **SQLite** - Base de datos única (checkpointer + configuración + auditoría)
- **FastAPI** - API HTTP async
- **structlog** - Logging estructurado JSON
- **pytest** - Testing

## Principios Arquitectónicos

1. **Structured Outputs obligatorios**: Cada agente LLM devuelve solo JSON validado con Pydantic
2. **Decisiones deterministas en código**: HITL y suficiencia de información son lógica Python pura, nunca delegada al LLM
3. **Configuración en BD**: Rúbricas, pesos y umbrales viven en SQLite, se inyectan vía prompt-templating
4. **Síntesis descriptiva únicamente**: El nodo sintetizador solo redacta narrativa; nunca calcula ni infiere cifras
5. **Logging estructurado en punto de decisión**: Cada decisión determinista se loggea donde ocurre

## Flujo del Grafo (Mermaid)

```mermaid
graph TD
    START --> INIT[init_state_node]
    INIT --> FEAS[feasibility_agent]
    INIT --> IMP[impact_agent]
    INIT --> COST[cost_agent]
    INIT --> NOV[novelty_agent]
    
    FEAS -->|missing fields| MISSING[missing_info_node]
    FEAS -->|continue| JOIN[join_agents]
    IMP -->|missing fields| MISSING
    IMP -->|continue| JOIN
    COST -->|missing fields| MISSING
    COST -->|continue| JOIN
    NOV -->|missing fields| MISSING
    NOV -->|continue| JOIN
    
    MISSING -->|retry same agent| FEAS
    MISSING -->|retry same agent| IMP
    MISSING -->|retry same agent| COST
    MISSING -->|retry same agent| NOV
    MISSING -->|all done| JOIN
    
    JOIN --> AGG[aggregator]
    AGG --> RISK[risk_gate]
    
    RISK -->|hitl_required| HITL[hitl_pause_node]
    RISK -->|no hitl| SYNTH[synthesizer_agent]
    
    HITL -->|approved| SYNTH
    HITL -->|rejected| HITL
    
    SYNTH --> END
```

## Estructura del Proyecto

```
src/proposal_evaluator/
├── schemas/          # Esquemas Pydantic (entrada, salida, estado del grafo)
├── config/           # Inicialización BD, repositorios de configuración
├── agents/           # 4 agentes evaluadores + sintetizador + gates deterministas
├── graph/            # Grafo LangGraph con fan-out paralelo + checkpointer SQLite
├── api/              # FastAPI endpoints (4 endpoints principales)
├── observability/    # Métricas y logging estructurado
└── llm/              # Capa de abstracción LLM (Anthropic)
```

## Instalación y Ejecución Local

### 1. Crear y activar entorno virtual

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2. Instalar dependencias
```bash
pip install -e ".[dev]"
```
*(Esto instala el paquete en modo editable junto con dependencias de desarrollo: ruff, mypy, pytest-asyncio)*

### 3. Configurar variables de entorno
```bash
cp .env.example .env
# Editar .env con tu ANTHROPIC_API_KEY
```

### 4. Inicializar base de datos
```bash
python -m proposal_evaluator.config.db_init
```

### 5. Correr tests
```bash
pytest
```

### 6. Iniciar servidor API
```bash
uvicorn proposal_evaluator.api.main:app --reload
```
La API estará disponible en `http://localhost:8000`
Documentación interactiva: `http://localhost:8000/docs`

## API Endpoints

### 1. Iniciar evaluación
```bash
curl -X POST http://localhost:8000/evaluations \
  -H "Content-Type: application/json" \
  -d '{
    "proposal_text": "Propuesta para implementar un sistema de IA generativa que automatice la generación de reportes financieros mensuales, reduciendo el tiempo manual de 20 horas a 2 horas.",
    "raw_proposal_data": {
      "descripcion_tecnica": "Uso de LLMs fine-tuned con RAG sobre datos financieros históricos",
      "recursos_necesarios": "GPU cluster, 3 ingenieros ML, datos históricos 5 años",
      "problema_que_resuelve": "Reportes manuales consumen 20h/semana del equipo financiero",
      "beneficio_esperado": "Ahorro 90% tiempo, consistencia, auditoría automática",
      "inversion_inicial_estimada": "150000",
      "descripcion_innovacion": "Pipeline RAG propietario con validación automática de alucinaciones",
      "diferenciadores_clave": "Datos privados + RAG + evaluación automática calidad"
    }
  }'
```

**Respuesta exitosa (201):**
```json
{
  "proposal_id": "abc123...",
  "status": "completed",
  "message": "Evaluación completada exitosamente",
  "state": {
    "proposal_id": "abc123...",
    "weighted_score": 77.5,
    "final_report": "=== REPORTE ... ===",
    ...
  }
}
```

### 2. Consultar estado de evaluación
```bash
curl http://localhost:8000/evaluations/abc123...
```

### 3. Proveer información faltante (si la evaluación está pausada)
```bash
curl -X POST http://localhost:8000/evaluations/abc123.../provide-info \
  -H "Content-Type: application/json" \
  -d '{
    "raw_proposal_data": {
      "descripcion_tecnica": "Información técnica añadida posteriormente",
      "recursos_necesarios": "Recursos actualizados"
    }
  }'
```

### 4. Aprobar/Rechazar HITL (validación humana por riesgo)
```bash
curl -X POST http://localhost:8000/evaluations/abc123.../hitl-approve \
  -H "Content-Type: application/json" \
  -d '{
    "approved": true,
    "comment": "Aprobado por comité directivo tras revisión de riesgos"
  }'
```

### 5. Métricas agregadas
```bash
curl http://localhost:8000/metrics/summary
```

**Respuesta:**
```json
{
  "total_evaluations_completed": 15,
  "hitl_triggered_count": 3,
  "hitl_rate_percent": 20.0,
  "missing_info_retries_count": 7,
  "retry_rate_percent": 46.67,
  "average_scores_by_criterion": {
    "feasibility": 72.3,
    "impact": 78.1,
    "cost": 65.4,
    "novelty": 71.2
  },
  "average_weighted_score": 71.75
}
```

## Health Check
```bash
curl http://localhost:8000/health
```

## Documentación Interactiva
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Variables de Entorno (.env)

```bash
# Requerido
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx

# Opcional
DATABASE_PATH=./data/proposal_evaluator.db
LLM_MODEL=claude-3-haiku-20240307
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2000
```

## Ejecutar Tests
```bash
# Todos los tests
pytest

# Solo tests de un módulo
pytest tests/test_graph.py -v

# Con cobertura
pytest --cov=proposal_evaluator tests/
```

## Arquitectura de Decisiones Clave

| Componente | Tipo | Responsabilidad |
|------------|------|-----------------|
| Completeness Gate | Determinista (Python) | Verifica campos requeridos vs raw_proposal_data |
| 4 Agentes Evaluadores | LLM (Structured Output) | Score 0-100 + summary + confidence |
| Aggregator | Determinista | Weighted score = Σ(score_i × weight_i) |
| Risk Gate | Determinista | Evalúa RiskRule activas → hitl_required |
| Missing Info Node | LangGraph interrupt | Pausa + reintentos (máx 3) → HITL |
| Risk HITL | LangGraph interrupt | Pausa para aprobación humana |
| Synthesizer | LLM (Narrativa only) | Texto cualitativo SIN números |
| Report Builder | Determinista (Python) | Template con números EXACTOS + narrativa |

## Configuración en Base de Datos (No hardcoded)

- **Rúbricas**: `rubrics` table (criterios, campos requeridos, escala_guía)
- **Pesos**: `criteria_weights` table (default 25% cada uno)
- **Reglas de riesgo**: `risk_rules` table (operador, umbral, descripción)
- **Eventos auditoría**: `audit_events` table (todos los hitos)

## Próximos Pasos Post-MVP (No implementados)

1. **Autenticación/Autorización**: JWT + roles (evaluator, approver, admin)
2. **Persistencia de reportes**: Almacenar reportes finales en BD/objetos
3. **Notificaciones async**: Webhooks/email para pausas HITL y completaciones
4. **Dashboard web**: Frontend para visualizar evaluaciones, métricas, historial
5. **Búsqueda semántica real**: Conectar novelty_agent a base de patentes (USPTO/EPO) + histórico Wazoku
6. **Versionado de rúbricas**: A/B testing de prompts y pesos
7. **Multi-tenant**: Aislamiento de datos por organización/cliente
8. **Rate limiting / Quotas**: Protección de API y costos LLM
9. **Observabilidad avanzada**: Traces distribuidos (OpenTelemetry), alertas
10. **CI/CD pipeline**: Tests automatizados, deploy staging/prod