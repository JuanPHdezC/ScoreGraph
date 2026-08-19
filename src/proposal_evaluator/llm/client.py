import json
import os
from typing import Type, TypeVar

import structlog
from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

from proposal_evaluator.schemas import AgentEvaluationOutput

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

MAX_RETRIES = 3


class LLMCallError(Exception):
    """Excepción lanzada cuando fallan todos los reintentos de llamada al LLM."""
    def __init__(self, message: str, last_error: Exception | None = None):
        super().__init__(message)
        self.last_error = last_error


def _schema_to_tool(schema_cls: Type[T]) -> dict:
    """Convierte un modelo Pydantic a formato tool de Anthropic."""
    schema = schema_cls.model_json_schema()
    return {
        "name": schema.get("title", schema_cls.__name__),
        "description": schema.get("description", "Structured output"),
        "input_schema": schema,
    }


def _build_correction_message(validation_error: ValidationError) -> str:
    """Construye un mensaje de corrección a partir de errores de validación Pydantic."""
    errors = []
    for err in validation_error.errors():
        loc = " -> ".join(str(x) for x in err["loc"])
        msg = err["msg"]
        errors.append(f"Campo '{loc}': {msg}")
    return (
        "La respuesta anterior no pasó la validación. Errores:\n"
        + "\n".join(f"- {e}" for e in errors)
        + "\n\nPor favor, responde SOLO con el JSON válido que cumpla el schema."
    )


def call_structured(
    system_prompt: str,
    user_message: str,
    output_schema: Type[T],
    max_retries: int = MAX_RETRIES,
) -> T:
    """
    Llama a Anthropic con structured output (tool-calling) y valida con Pydantic.
    Reintenta hasta max_retries veces si falla parseo/validación.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMCallError("ANTHROPIC_API_KEY no configurada en variables de entorno")

    client = Anthropic(api_key=api_key)
    tool = _schema_to_tool(output_schema)

    messages = [{"role": "user", "content": user_message}]

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(
                "llm_call_attempt",
                attempt=attempt,
                max_retries=max_retries,
                schema=output_schema.__name__,
            )

            response = client.messages.create(
                model=os.getenv("LLM_MODEL", "claude-3-haiku-20240307"),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
                system=system_prompt,
                messages=messages,
                tools=[tool],
                tool_choice={"type": "tool", "name": tool["name"]},
            )

            # Extraer el tool_use
            tool_use = next(
                (block for block in response.content if block.type == "tool_use"),
                None,
            )
            if not tool_use:
                raise LLMCallError("Respuesta sin tool_use")

            # Parsear y validar con Pydantic
            result = output_schema.model_validate(tool_use.input)
            logger.info("llm_call_success", schema=output_schema.__name__, attempt=attempt)
            return result

        except ValidationError as e:
            logger.warning("llm_validation_failed", attempt=attempt, errors=e.errors())
            if attempt == max_retries:
                raise LLMCallError(
                    f"Validación falló tras {max_retries} intentos", e
                ) from e
            # Añadir mensaje de corrección y reintentar
            correction = _build_correction_message(e)
            messages.append({"role": "assistant", "content": str(response.content)})
            messages.append({"role": "user", "content": correction})

        except Exception as e:
            logger.error("llm_call_error", attempt=attempt, error=str(e))
            if attempt == max_retries:
                raise LLMCallError(f"Error en llamada LLM tras {max_retries} intentos", e) from e

    raise LLMCallError(f"Se agotaron {max_retries} reintentos")