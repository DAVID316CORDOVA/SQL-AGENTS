"""
agents/base_skill_agent.py

Clase base compartida por todos los agentes del sistema (AR, APS, AG, AV, AE, AS).

Implementa el patron ReAct (Reasoning + Acting): en cada iteracion el LLM
decide si necesita invocar una skill (herramienta determinista) o si ya
tiene suficiente informacion para producir su respuesta final en JSON.
Este ciclo se repite hasta que el LLM responde sin invocar skills o se
alcanza el limite de iteraciones.

Soporta dos proveedores LLM con APIs distintas:
  - OpenAI  (gpt-4o, gpt-4o-mini, gemini via openai-compat)
  - Anthropic (claude-haiku, claude-sonnet, claude-opus)
El proveedor se detecta por el NOMBRE del modelo, no por config global,
lo que permite mezclar modelos de distintos proveedores en el mismo experimento.
"""

import json
import re
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client import get_client, get_provider, call_llm, build_tool_result_message
from llm_client import _detect_provider_from_model, build_observability_kwargs


def _extract_json(text: str) -> str:
    # Algunos modelos (gemini, claude) envuelven el JSON en bloques ```json ... ```.
    # Esta funcion extrae solo el contenido del bloque para que json.loads no falle.
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        extracted = match.group(1).strip()
        if extracted:
            return extracted
    return text.strip()


def _openai_tools_to_anthropic(tools: list) -> list:
    # Convierte la definicion de skills del formato OpenAI (function.parameters)
    # al formato Anthropic (input_schema), que es el que usa el loop interno.
    # Permite que todos los agentes definan sus skills una sola vez y el
    # base agent las adapte al proveedor correcto en tiempo de ejecucion.
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


class SkillAgent:
    """
    Clase base para agentes habilitados con skills (tool use).

    Las subclases deben:
      - Definir self.client y self.model en __init__ (via llm_client.get_client())
      - Implementar _get_skills() con los schemas de sus skills
      - Implementar _execute_skill(name, args) para despachar cada skill
    """

    try:
        from config import MAX_SKILL_ITERATIONS as _cfg_iter
        MAX_SKILL_ITERATIONS = _cfg_iter
    except ImportError:
        # Limite de iteraciones del loop ReAct por defecto. Cada iteracion
        # = una llamada al LLM + N ejecuciones de skills. Aumentar si se
        # necesita encadenamiento de muchas skills en consultas complejas.
        MAX_SKILL_ITERATIONS = 8

    VERBOSE = False

    # ToolRegistry opcional: controla que skills puede invocar cada agente.
    # Si no esta seteado, no hay enforcement (todos los agentes pueden usar
    # cualquier skill que tengan definida).
    _registry = None
    _registry_role = None

    def attach_registry(self, registry, role: str) -> None:
        """Asocia un ToolRegistry al agente para aplicar control de acceso por skill."""
        self._registry = registry
        self._registry_role = role

    def _get_skills(self) -> list:
        # Las subclases sobreescriben este metodo para exponer sus skills al LLM.
        # Si retorna lista vacia, el agente funciona como chat puro (sin tool use).
        return []

    def _execute_skill(self, skill_name: str, args: dict) -> str:
        # Las subclases sobreescriben este metodo para despachar cada skill
        # a su implementacion concreta (funcion Python determinista).
        return json.dumps({"error": f"Skill '{skill_name}' no implementada en este agente"})

    def _dispatch_skill_safely(self, skill_name: str, args: dict) -> str:
        # Wrapper de seguridad: verifica permisos del ToolRegistry antes de ejecutar.
        # Si el registry no esta configurado, llama directamente a _execute_skill.
        # Esto permite que el mismo codigo funcione con y sin enforcement de permisos.
        registry = self._registry
        role = self._registry_role
        if registry is not None and role:
            if not registry.can_use(role, skill_name):
                allowed = registry._agent_permissions.get(role, [])
                registry._execution_log.append({
                    "agent":  role,
                    "tool":   skill_name,
                    "status": "DENIED",
                    "reason": f"Skill '{skill_name}' no autorizada para '{role}'. Permitidas: {allowed}",
                })
                return json.dumps({
                    "error": f"PERMISSION_DENIED: agent '{role}' cannot invoke "
                             f"skill '{skill_name}'. Allowed: {allowed}"
                })
            registry._execution_log.append({
                "agent":  role,
                "tool":   skill_name,
                "status": "ALLOWED",
                "kwargs_keys": list(args.keys()) if isinstance(args, dict) else [],
            })
        return self._execute_skill(skill_name, args)

    def _run_agent_loop(self, system_prompt: str, user_message: str,
                        temperature: float = 0.1,
                        top_p: float | None = None,
                        json_output: bool = True,
                        force_tools: bool = False) -> str:
        # Punto de entrada del ciclo ReAct. Convierte las skills al formato
        # neutral (Anthropic) y delega al loop correcto segun el proveedor.
        raw_skills = self._get_skills()
        tools = _openai_tools_to_anthropic(raw_skills)

        agent_label = getattr(self, "_agent_label", self.__class__.__name__)
        verbose = SkillAgent.VERBOSE and bool(tools)

        if verbose:
            skill_names = [t["name"] for t in tools]
            print(f"\n  [SKILLS - {agent_label}] Skills disponibles: {', '.join(skill_names)}")

        effective_system = system_prompt
        if json_output:
            # Refuerzo explicito: algunos modelos ignoran la instruccion JSON del prompt
            # cuando tienen acceso a skills. Forzarlo en el system asegura el formato.
            effective_system = (
                system_prompt.rstrip()
                + "\n\nIMPORTANTE: Responde SIEMPRE con JSON valido."
            )

        # La deteccion del proveedor se hace por NOMBRE DE MODELO para permitir
        # experimentos cross-proveedor (ej. Optuna probando gpt-4o y claude-sonnet
        # en el mismo trial sin cambiar LLM_PROVIDER en config).
        provider = _detect_provider_from_model(self.model)

        if provider == "anthropic":
            return self._loop_anthropic(effective_system, user_message, tools, temperature, json_output, verbose, agent_label, force_tools, top_p=top_p)
        else:
            return self._loop_openai(effective_system, user_message, tools, temperature, json_output, verbose, agent_label, force_tools, top_p=top_p)

    def _loop_openai(self, system_prompt, user_message, tools, temperature, json_output, verbose, agent_label, force_tools=False, top_p=None):
        # Historial de mensajes acumulativo: cada iteracion agrega la respuesta
        # del asistente y los resultados de skills, formando el contexto del loop.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Convertir skills del formato Anthropic neutral al formato OpenAI
        openai_tools = None
        if tools:
            openai_tools = []
            for t in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                    }
                })

        # Ciclo ReAct: continua hasta que el LLM responde sin invocar skills
        # o se alcanza MAX_SKILL_ITERATIONS
        for iteration in range(self.MAX_SKILL_ITERATIONS):
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 4096,
            }
            if top_p is not None:
                kwargs["top_p"] = top_p
            kwargs.update(build_observability_kwargs(self.model, name=agent_label))
            if openai_tools:
                kwargs["tools"] = openai_tools
                # Solo en la primera iteracion se fuerza el uso de skills (force_tools=True).
                # En las siguientes el LLM decide libremente si invocar o responder.
                kwargs["tool_choice"] = "required" if (force_tools and iteration == 0) else "auto"

            # Reintentos con backoff exponencial para errores transitorios de la API
            # (rate limit 429, errores de servidor 5xx). Hasta 4 intentos por llamada.
            for _attempt in range(4):
                try:
                    response = self.client.chat.completions.create(**kwargs)
                    break
                except Exception as _e:
                    code = getattr(getattr(_e, "response", None), "status_code", None) or getattr(_e, "status_code", None)
                    if code in (429, 500, 502, 503, 529) and _attempt < 3:
                        time.sleep(10 * (2 ** _attempt))
                    else:
                        raise
            choice = response.choices[0]
            message = choice.message

            # Guardar la respuesta del asistente en el historial antes de procesar skills.
            # Es obligatorio para que la API acepte el mensaje "tool" subsiguiente.
            assistant_msg = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ]
            messages.append(assistant_msg)

            if not message.tool_calls:
                # El LLM no invoco skills: tiene su respuesta final
                if verbose:
                    print(f"    Iter {iteration + 1}: LLM responde sin invocar skills → RESPUESTA FINAL")
                # Gemini 2.5 usa "thinking": message.content puede ser None aunque
                # el texto real este en reasoning_content (campo no estandar).
                if not message.content and "gemini-2.5" in self.model:
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
                raw = message.content or "{}"
                return _extract_json(raw) if json_output else raw

            if verbose:
                print(f"    Iter {iteration + 1}: LLM invoca {len(message.tool_calls)} skill(s)")

            # Ejecutar cada skill solicitada por el LLM y agregar los resultados al historial.
            # El LLM usara estos resultados en la siguiente iteracion para razonar.
            for tc in message.tool_calls:
                func_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                if verbose:
                    args_str = json.dumps(args, ensure_ascii=False)
                    args_preview = args_str[:120] + "..." if len(args_str) > 120 else args_str
                    print(f"      > {func_name}({args_preview})")

                try:
                    skill_result = self._dispatch_skill_safely(func_name, args)
                except Exception as exc:
                    skill_result = json.dumps({"error": str(exc)})

                if verbose:
                    self._print_skill_result(skill_result)

                # El resultado de cada skill se agrega como mensaje "tool" al historial
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": skill_result,
                })

        if verbose:
            print(f"    Limite de iteraciones ({self.MAX_SKILL_ITERATIONS}) alcanzado → forzando respuesta")

        # Se agoto el limite: forzar respuesta final con todo el contexto acumulado
        messages.append({
            "role": "user",
            "content": (
                "Has alcanzado el limite de iteraciones. "
                "Proporciona tu respuesta final en JSON con todos los "
                "resultados que obtuviste de las skills."
            )
        })
        final_response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=messages,
        )
        final_msg = final_response.choices[0].message
        if not final_msg.content and "gemini-2.5" in self.model:
            try:
                raw_dict = final_response.model_dump()
                msg_dict = raw_dict.get("choices", [{}])[0].get("message", {})
                for field in ["content", "reasoning_content"]:
                    val = msg_dict.get(field) or ""
                    if isinstance(val, str) and val.strip():
                        final_msg.content = val
                        break
            except Exception:
                pass
        raw = final_msg.content or "{}"
        return _extract_json(raw) if json_output else raw

    def _loop_anthropic(self, system_prompt, user_message, tools, temperature, json_output, verbose, agent_label, force_tools=False, top_p=None):
        # La API de Anthropic recibe el system prompt separado (no en messages[])
        messages = [{"role": "user", "content": user_message}]

        # Ciclo ReAct identico al de OpenAI pero usando la API de Anthropic Messages
        for iteration in range(self.MAX_SKILL_ITERATIONS):
            call_kwargs = {
                "model": self.model,
                "max_tokens": 4096,
                "system": system_prompt,
                "messages": messages,
                "temperature": temperature,
            }
            # Anthropic rechaza la combinacion temperature != 1.0 + top_p.
            # Solo se pasa top_p cuando temperature esta en su valor default.
            if top_p is not None and temperature == 1.0:
                call_kwargs["top_p"] = top_p
            call_kwargs.update(build_observability_kwargs(self.model, name=agent_label))
            if tools:
                call_kwargs["tools"] = tools
                call_kwargs["tool_choice"] = (
                    {"type": "any"} if (force_tools and iteration == 0)
                    else {"type": "auto"}
                )

            response = self.client.messages.create(**call_kwargs)
            # Agregar la respuesta completa del asistente (puede contener bloques
            # de texto Y bloques tool_use mezclados en el mismo mensaje)
            messages.append({"role": "assistant", "content": response.content})

            # Anthropic separa los bloques de texto y tool_use en content[]
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                # Sin bloques tool_use: el LLM produce su respuesta final
                if verbose:
                    print(f"    Iter {iteration + 1}: LLM responde sin invocar skills → RESPUESTA FINAL")
                text_blocks = [b for b in response.content if b.type == "text"]
                raw = text_blocks[0].text if text_blocks else "{}"
                return _extract_json(raw) if json_output else raw

            if verbose:
                print(f"    Iter {iteration + 1}: LLM invoca {len(tool_use_blocks)} skill(s)")

            # Ejecutar todas las skills del turno y agrupar resultados en un
            # unico mensaje "user" con type "tool_result" (formato Anthropic)
            tool_results = []
            for block in tool_use_blocks:
                if verbose:
                    args_str = json.dumps(block.input, ensure_ascii=False)
                    args_preview = args_str[:120] + "..." if len(args_str) > 120 else args_str
                    print(f"      > {block.name}({args_preview})")

                try:
                    skill_result = self._dispatch_skill_safely(block.name, block.input)
                except Exception as exc:
                    skill_result = json.dumps({"error": str(exc)})

                if verbose:
                    self._print_skill_result(skill_result)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": skill_result,
                })

            # Todos los resultados del turno se devuelven como un solo mensaje user
            messages.append({"role": "user", "content": tool_results})

        if verbose:
            print(f"    Limite de iteraciones ({self.MAX_SKILL_ITERATIONS}) alcanzado → forzando respuesta")

        # Mismo mecanismo de forzado de respuesta que en OpenAI
        messages.append({
            "role": "user",
            "content": (
                "Has alcanzado el limite de iteraciones. "
                "Proporciona tu respuesta final en JSON con todos los "
                "resultados que obtuviste de las skills."
            )
        })
        final_response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
        )
        text_blocks = [b for b in final_response.content if b.type == "text"]
        raw = text_blocks[0].text if text_blocks else "{}"
        return _extract_json(raw) if json_output else raw

    def _print_skill_result(self, skill_result):
        try:
            r = json.loads(skill_result)
            if "errors" in r or "warnings" in r:
                errs = len(r.get("errors", []))
                warns = len(r.get("warnings", []))
                print(f"      < errors={errs} | warnings={warns}")
            elif "estimated_rows" in r:
                rows = r.get("estimated_rows", "?")
                scan = r.get("has_full_scan", False)
                time_ms = r.get("actual_time_ms")
                t_str = f" | tiempo={time_ms:.1f}ms" if time_ms is not None else ""
                print(f"      < rows={rows} | full_scan={scan}{t_str}")
            elif "tables" in r or "columns" in r:
                tables = list(r.get("tables", {}).keys()) if isinstance(r.get("tables"), dict) else []
                cols = len(r.get("columns", []))
                print(f"      < tablas={tables} | columnas={cols}")
            elif "sql_formatted" in r:
                lines = r["sql_formatted"].count("\n") + 1
                print(f"      < SQL formateado ({lines} lineas)")
            elif "error" in r:
                print(f"      < ERROR: {r['error'][:80]}")
            else:
                preview = skill_result[:100] + "..." if len(skill_result) > 100 else skill_result
                print(f"      < {preview}")
        except Exception:
            preview = skill_result[:100] + "..." if len(skill_result) > 100 else skill_result
            print(f"      < {preview}")
