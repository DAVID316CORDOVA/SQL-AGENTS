"""
llm_client.py

Abstraccion del proveedor LLM. Por defecto usa OpenAI.
Si LLM_PROVIDER="anthropic" en config.py, usa Anthropic.

Uso:
    from llm_client import get_client, call_llm, get_model, get_provider

Ubicacion: llm_client.py (raiz)
"""

import os
import json
import re

try:
    from config import LLM_PROVIDER
except ImportError:
    LLM_PROVIDER = "openai"

# Mapeo de modelos Anthropic → OpenAI equivalentes
_MODEL_MAP_TO_OPENAI = {
    "claude-haiku-4-5": "gpt-4o-mini",
    "claude-sonnet-4-6": "gpt-4o",
    "claude-opus-4-6": "gpt-4o",
}

_MODEL_MAP_TO_ANTHROPIC = {v: k for k, v in _MODEL_MAP_TO_OPENAI.items()}


# ────────────────────────────────────────────────────────────────────
# Backend Anthropic: "api" (default) o "bedrock"
# ────────────────────────────────────────────────────────────────────
def _anthropic_backend() -> str:
    return os.getenv("ANTHROPIC_BACKEND", "api").lower()


# Mapeo de nombres lógicos (los que ya usan los agentes) -> inference
# profile IDs de Bedrock. Se sobreescriben por env vars BEDROCK_MODEL_*
# si están definidas. Esto deja un solo punto de actualización cuando
# Anthropic publica una nueva versión.
_BEDROCK_PROFILE_DEFAULTS = {
    "claude-haiku-4-5":  "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4-5": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-sonnet-4-6": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-opus-4-5":   "global.anthropic.claude-opus-4-5-20251101-v1:0",
    "claude-opus-4-6":   "global.anthropic.claude-opus-4-5-20251101-v1:0",
}

_BEDROCK_ENV_KEYS = {
    "claude-haiku-4-5":  "BEDROCK_MODEL_HAIKU_4_5",
    "claude-sonnet-4-5": "BEDROCK_MODEL_SONNET_4_5",
    "claude-sonnet-4-6": "BEDROCK_MODEL_SONNET_4_5",
    "claude-opus-4-5":   "BEDROCK_MODEL_OPUS_4_5",
    "claude-opus-4-6":   "BEDROCK_MODEL_OPUS_4_5",
}

# Llama via Bedrock - mapeo de nombres logicos a inference profile IDs.
# Sobreescribible via env vars BEDROCK_MODEL_LLAMA_*.
_BEDROCK_LLAMA_PROFILES = {
    "llama-3-3-70b":     "us.meta.llama3-3-70b-instruct-v1:0",
    "llama-3.3-70b":     "us.meta.llama3-3-70b-instruct-v1:0",
    "meta.llama3-3-70b": "us.meta.llama3-3-70b-instruct-v1:0",
    "llama-3-1-70b":     "us.meta.llama3-1-70b-instruct-v1:0",
    "llama-3-1-8b":      "us.meta.llama3-1-8b-instruct-v1:0",
}


def _resolve_llama_profile(model: str) -> str:
    """Resuelve nombre logico Llama -> inference profile ID de Bedrock."""
    env_key = f"BEDROCK_MODEL_{model.upper().replace('-', '_').replace('.', '_')}"
    if os.getenv(env_key):
        return os.getenv(env_key)
    if model in _BEDROCK_LLAMA_PROFILES:
        return _BEDROCK_LLAMA_PROFILES[model]
    # Si ya viene con prefijo meta./us.meta., pasarlo tal cual
    if model.startswith("meta.") or model.startswith("us.meta."):
        return model
    return model


def _build_bedrock_runtime_client():
    """Cliente boto3 bedrock-runtime usando AWS_PROFILE o credenciales por defecto."""
    import boto3
    region = os.getenv("AWS_REGION") or os.getenv("BEDROCK_REGION") or "us-east-1"
    profile = os.getenv("AWS_PROFILE")
    if profile:
        session = boto3.Session(profile_name=profile, region_name=region)
        return session.client("bedrock-runtime")
    return boto3.client("bedrock-runtime", region_name=region)


def _resolve_bedrock_profile_id(logical_model: str) -> str:
    """Resuelve el inference profile ID de Bedrock para un modelo lógico."""
    env_key = _BEDROCK_ENV_KEYS.get(logical_model)
    if env_key and os.getenv(env_key):
        return os.getenv(env_key)
    if logical_model in _BEDROCK_PROFILE_DEFAULTS:
        return _BEDROCK_PROFILE_DEFAULTS[logical_model]
    return logical_model


def _build_anthropic_bedrock_client():
    """Construye un cliente AnthropicBedrock con credenciales del perfil AWS."""
    import anthropic
    profile = os.getenv("AWS_PROFILE")
    region  = os.getenv("AWS_REGION") or os.getenv("BEDROCK_REGION") or "us-east-1"
    if profile:
        import boto3
        session = boto3.Session(profile_name=profile)
        creds = session.get_credentials()
        if creds is None:
            raise ValueError(f"AWS_PROFILE='{profile}' sin credenciales válidas")
        frozen = creds.get_frozen_credentials()
        return anthropic.AnthropicBedrock(
            aws_access_key=frozen.access_key,
            aws_secret_key=frozen.secret_key,
            aws_session_token=frozen.token,
            aws_region=region,
        )
    return anthropic.AnthropicBedrock(aws_region=region)


def get_provider() -> str:
    return LLM_PROVIDER.lower()


def get_client():
    provider = get_provider()
    if provider == "anthropic":
        if _anthropic_backend() == "bedrock":
            return _build_anthropic_bedrock_client()
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY no configurada")
        return anthropic.Anthropic(api_key=api_key)
    else:
        import openai
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY no configurada. Configurala en config.py")
        return openai.OpenAI(api_key=api_key)


# ─────────────────────────────────────────────────────────────────────
# Observability: Langfuse (drop-in OpenAI) + LangSmith (wrappers)
# Activacion automatica si las API keys correspondientes existen en .env.
# Opt-out: ENABLE_OBSERVABILITY=0 desactiva ambos sin tocar el codigo.
# ─────────────────────────────────────────────────────────────────────

def _observability_enabled() -> bool:
    return os.getenv("ENABLE_OBSERVABILITY", "1").lower() not in ("0", "false", "no")


def _langfuse_active() -> bool:
    return _observability_enabled() and bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    )


def _langsmith_active() -> bool:
    return _observability_enabled() and bool(os.getenv("LANGSMITH_API_KEY"))


def _get_openai_class():
    """OpenAI class — Langfuse-wrapped (drop-in) si observability esta activa."""
    if _langfuse_active():
        try:
            from langfuse.openai import OpenAI as LfOpenAI
            return LfOpenAI
        except ImportError:
            pass
    import openai
    return openai.OpenAI


def _provider_short_name(model: str) -> str:
    """Nombre corto del proveedor para tags / project routing."""
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gemini-"):
        return "gemini"
    if model.startswith(("gpt-", "o1", "o3")):
        return "openai"
    if model.startswith(("llama-", "meta.", "us.meta.")):
        return "bedrock-llama"
    return "unknown"


def build_observability_kwargs(model: str, name: str = "llm-call") -> dict:
    """
    Construye los kwargs a inyectar en client.create() para que LangSmith
    capture el trace en el proyecto correcto por proveedor.

    LangSmith: rutea a 3 proyectos separados
        sql-agents-openai / sql-agents-anthropic / sql-agents-gemini.
        wrap_openai / wrap_anthropic consumen `langsmith_extra` antes de
        pasar la llamada al SDK subyacente.

    Langfuse: NO se inyectan kwargs aqui. Cuando LangSmith envuelve al
        cliente langfuse, los kwargs Langfuse (tags/name/metadata)
        atraviesan el chain y rompen el SDK de OpenAI. Langfuse igual
        captura todo automaticamente y permite filtrar por `model` en
        la UI, lo que equivale a separacion por proveedor (gpt-* vs
        claude-* vs gemini-*).

    Devuelve {} si no hay observability activa.
    """
    if not _langsmith_active():
        return {}
    provider = _provider_short_name(model)
    return {
        "langsmith_extra": {
            "project_name": f"sql-agents-{provider}",
            "metadata":     {"provider": provider, "model_name": model},
            "tags":         [provider, model],
            "run_name":     f"{name}-{provider}",
        }
    }


def flush_observability() -> None:
    """Asegura que todos los traces async de Langfuse se envien antes de salir."""
    if _langfuse_active():
        try:
            from langfuse import get_client
            get_client().flush()
        except Exception:
            pass


def _maybe_wrap_langsmith(client, model: str):
    """Wrap el cliente con LangSmith tracing si esta activo y disponible."""
    if not _langsmith_active():
        return client
    # Asegurar que LangSmith use el project configurado
    os.environ.setdefault(
        "LANGCHAIN_PROJECT",
        os.getenv("LANGSMITH_PROJECT", "sql-agents-tesis"),
    )
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    try:
        from langsmith.wrappers import wrap_openai, wrap_anthropic
        if model.startswith("claude-"):
            return wrap_anthropic(client)
        # gemini, gpt y fallback usan SDK OpenAI-compat
        return wrap_openai(client)
    except ImportError:
        return client


def get_client_for_model(model: str):
    """
    Devuelve el cliente apropiado segun el nombre del modelo.

    Routing por prefijo del nombre del modelo (sobreescribe LLM_PROVIDER):
      - "gemini-..."  -> OpenAI SDK apuntando al endpoint OpenAI-compatible
                         de Google AI Studio (GEMINI_API_KEY).
      - "claude-..."  -> Anthropic SDK directo (ANTHROPIC_API_KEY).
      - "gpt-..."     -> OpenAI SDK regular (OPENAI_API_KEY).
      - otro          -> fallback a get_client() (segun LLM_PROVIDER).

    Si las API keys de Langfuse y/o LangSmith estan en .env (y
    ENABLE_OBSERVABILITY no esta en "0"), el cliente queda instrumentado
    para enviar traces a ambos dashboards automaticamente.
    """
    if model.startswith("gemini-"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY no configurada en .env")
        OpenAIClass = _get_openai_class()
        client = OpenAIClass(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        return _maybe_wrap_langsmith(client, model)
    if model.startswith("llama-") or model.startswith("meta.") or model.startswith("us.meta."):
        return _build_bedrock_runtime_client()
    if model.startswith("claude-"):
        if _anthropic_backend() == "bedrock":
            return _build_anthropic_bedrock_client()
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY no configurada en .env")
        client = anthropic.Anthropic(api_key=api_key)
        return _maybe_wrap_langsmith(client, model)
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY no configurada")
        OpenAIClass = _get_openai_class()
        client = OpenAIClass(api_key=api_key)
        return _maybe_wrap_langsmith(client, model)
    return get_client()


def get_model(agent_name: str, backend: str | None = None,
              dataset: str | None = None) -> str:
    """
    Devuelve el modelo para un agente. Si se pasa `backend` y/o `dataset`,
    busca primero en el lookup detallado (con fallback en cascada via
    config.get_agent_model). Si no, usa el AGENT_MODELS plano.

    Backward compatible: get_model("AR") sigue funcionando.
    """
    try:
        from config import get_agent_model
        return get_agent_model(agent_name, backend, dataset)
    except ImportError:
        try:
            from config import AGENT_MODELS
            model = AGENT_MODELS.get(agent_name)
            if model:
                return model
        except ImportError:
            pass
    provider = get_provider()
    if provider == "openai":
        return "gpt-4o-mini"
    return "claude-haiku-4-5"


def get_temperature(agent_name: str, backend: str | None = None,
                    dataset: str | None = None,
                    default: float = 0.1) -> float:
    """Idem get_model pero para la temperatura del agente."""
    try:
        from config import get_agent_temperature
        return get_agent_temperature(agent_name, backend, dataset, default=default)
    except ImportError:
        return default


def _detect_provider_from_model(model: str) -> str:
    """Detecta el proveedor real por el prefijo del modelo (override de LLM_PROVIDER)."""
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("llama-") or model.startswith("meta.") or model.startswith("us.meta."):
        return "bedrock_llama"
    if model.startswith("gemini-") or model.startswith("gpt-") \
            or model.startswith("o1") or model.startswith("o3"):
        return "openai"  # Gemini OpenAI-compat usa el mismo path que OpenAI
    return get_provider()  # fallback al config global


def call_llm(client, model: str, messages: list,
             system: str = None, temperature: float = 0.1,
             max_tokens: int = 4096, tools: list = None) -> dict:
    """
    Llamada unificada al LLM. Retorna dict con:
      - content: str (texto de respuesta)
      - tool_calls: list[dict] con {id, name, arguments} o [] si no hay
      - raw: respuesta original del proveedor
      - _assistant_message: mensaje para agregar al historial

    El proveedor se detecta del nombre del modelo (claude-* -> anthropic,
    gpt-*/gemini-* -> openai). Asi Optuna puede mezclar proveedores en
    el mismo experimento sin tocar LLM_PROVIDER global.
    """
    provider = _detect_provider_from_model(model)
    if provider == "anthropic":
        return _call_anthropic(client, model, messages, system, temperature, max_tokens, tools)
    if provider == "bedrock_llama":
        return _call_bedrock_llama(client, model, messages, system, temperature, max_tokens, tools)
    return _call_openai(client, model, messages, system, temperature, max_tokens, tools)


def _call_openai(client, model, messages, system, temperature, max_tokens, tools):
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})

    for msg in messages:
        if msg["role"] == "user" and isinstance(msg["content"], list):
            # Tool results de Anthropic → convertir a mensajes tool de OpenAI
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "tool_result":
                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": item["tool_use_id"],
                        "content": item["content"],
                    })
        elif msg["role"] == "assistant" and isinstance(msg["content"], list):
            # Content blocks de Anthropic → convertir a formato OpenAI
            text_parts = []
            openai_tool_calls = []
            for block in msg["content"]:
                if hasattr(block, "type"):
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        openai_tool_calls.append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input),
                            }
                        })
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        openai_tool_calls.append({
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block.get("input", {})),
                            }
                        })
            assistant_msg = {"role": "assistant", "content": "\n".join(text_parts) or None}
            if openai_tool_calls:
                assistant_msg["tool_calls"] = openai_tool_calls
            full_messages.append(assistant_msg)
        else:
            full_messages.append(msg)

    kwargs = {
        "model": model,
        "messages": full_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if tools:
        openai_tools = _to_openai_tools(tools)
        kwargs["tools"] = openai_tools
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    message = choice.message

    # Gemini-2.5 con thinking: message.content puede ser None aunque haya
    # texto en campos no estándar. Intentamos extraerlo del dict raw.
    if not message.content and "gemini-2.5" in model:
        try:
            raw_dict = response.model_dump()
            msg_dict = raw_dict.get("choices", [{}])[0].get("message", {})
            for field in ["content", "reasoning_content"]:
                val = msg_dict.get(field) or ""
                if isinstance(val, str) and val.strip():
                    message.content = val
                    break
        except Exception:
            pass

    tool_calls = []
    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": args,
            })

    return {
        "content": message.content or "",
        "tool_calls": tool_calls,
        "raw": response,
        "_assistant_message": message,
    }


def _call_anthropic(client, model, messages, system, temperature, max_tokens, tools):
    # Auto-convert OpenAI model names to Anthropic equivalents
    model = _MODEL_MAP_TO_ANTHROPIC.get(model, model)
    # Si vamos por Bedrock, traducir nombre lógico a inference profile ID
    if _anthropic_backend() == "bedrock":
        model = _resolve_bedrock_profile_id(model)
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "temperature": temperature,
    }
    if system:
        kwargs["system"] = system
    if tools:
        anthropic_tools = _to_anthropic_tools(tools)
        kwargs["tools"] = anthropic_tools
        kwargs["tool_choice"] = {"type": "auto"}

    response = client.messages.create(**kwargs)

    text_parts = []
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "name": block.name,
                "arguments": block.input,
            })

    return {
        "content": "\n".join(text_parts),
        "tool_calls": tool_calls,
        "raw": response,
        "_assistant_message": response.content,
    }


def _call_bedrock_llama(client, model, messages, system, temperature, max_tokens, tools):
    """
    Llamada a Llama 3.x via Bedrock usando la Converse API (la API unificada
    que soporta inference profiles cross-region us.meta.*). invoke_model
    falla con "Operation not allowed" para inference profiles.

    Tool use: se ignoran los `tools` aqui — Llama via Bedrock no soporta
    function calling estructurado en este path (devolvemos tool_calls=[]).
    """
    profile_id = _resolve_llama_profile(model)

    converse_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)
        converse_messages.append({
            "role": role,
            "content": [{"text": str(content)}],
        })

    kwargs = {
        "modelId":  profile_id,
        "messages": converse_messages,
        "inferenceConfig": {
            "maxTokens":   max_tokens,
            "temperature": temperature,
        },
    }
    if system:
        kwargs["system"] = [{"text": system}]

    response = client.converse(**kwargs)
    out_msg  = response["output"]["message"]
    text     = "".join(
        block.get("text", "") for block in out_msg.get("content", [])
    )

    return {
        "content":    text,
        "tool_calls": [],
        "raw":        response,
        "_assistant_message": {"role": "assistant", "content": text},
    }


def build_tool_result_message(tool_call_id: str, content: str) -> dict:
    provider = get_provider()
    if provider == "anthropic":
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": content,
        }
    else:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }


def build_tool_results_as_user_message(results: list) -> dict:
    """Para Anthropic, los tool results van como content de un user message."""
    provider = get_provider()
    if provider == "anthropic":
        return {"role": "user", "content": results}
    else:
        return results  # OpenAI: cada result es un mensaje independiente


def build_assistant_message(raw_response) -> dict:
    """Construye el mensaje assistant para agregar al historial."""
    provider = get_provider()
    if provider == "anthropic":
        return {"role": "assistant", "content": raw_response}
    else:
        return {"role": "assistant", "content": raw_response.content,
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.function.name,
                                             "arguments": tc.function.arguments}}
                               for tc in (raw_response.tool_calls or [])]}


def _to_openai_tools(tools: list) -> list:
    openai_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            openai_tools.append(tool)
        else:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                }
            })
    return openai_tools


def _to_anthropic_tools(tools: list) -> list:
    anthropic_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool["function"]
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        else:
            anthropic_tools.append(tool)
    return anthropic_tools
