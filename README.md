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

## Estructura del Proyecto

```
src/proposal_evaluator/
├── schemas/          # Esquemas Pydantic (entrada, salida, estado del grafo)
├── config/           # Inicialización BD, repositorios de configuración
├── agents/           # Agentes evaluadores (fase 2)
├── graph/            # Grafo LangGraph (fase 2)
├── api/              # FastAPI endpoints (fase 3)
└── llm/              # Capa de abstracción LLM (fase 2)
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

### 6. Iniciar servidor API (fase 3+)
```bash
uvicorn proposal_evaluator.api.main:app --reload
```

## API Endpoints (previstos)

- `POST /evaluations` - Iniciar evaluación de una propuesta
- `GET /evaluations/{id}` - Obtener estado/resultado
- `POST /evaluations/{id}/hitl` - Resolver HITL y continuar
- `GET /config/rubrics` - Listar rúbricas
- `GET /config/risk-rules` - Listar reglas de riesgo