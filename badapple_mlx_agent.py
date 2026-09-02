"""Bad Apple MLX server — agent protocol handler and multi-step task management.

Extracted from badapple_mlx_server.py. Contains the LAP (Local Agent Protocol)
request dispatcher, the autonomous plan/act/observe agent loop, and background
agent task lifecycle management.
"""
import asyncio
import json
import os
import re
import time
import traceback
from typing import Any

import mlx.core as mx
import badapple_agent_tasks
import badapple_ambient
import badapple_identity
import badapple_ocular
import badapple_p2p
from badapple_mlx_tools import TOOLS, extract_tool_calls, postprocess_output, tools_for_prompt

async def _write_frame(writer: asyncio.StreamWriter, frame: dict):
    data = json.dumps(frame).encode() + b"\n"
    writer.write(data)
    await writer.drain()


PLANNER_SYSTEM_PROMPT = (
    "You are a task planner. The user wants a multi-step local action completed. "
    "Break the task into 1-4 short steps. For each step output exactly one line in this format:\n"
    "TOOL:<tool_name>:<json_arguments>\n"
    "or\n"
    "SAY:<what the assistant should tell the user after the previous tool results>\n"
    "Available tools:\n"
    '- list_directory: {"path": "..."}\n'
    '- read_file: {"path": "...", "limit": 5000}\n'
    '- search_content: {"query": "...", "path": "...", "max_results": 20}\n'
    '- run_shell: {"command": "..."}\n'
    '- write_file: {"filename": "...", "content": "...", "append": false}\n'
    '- run_applescript: {"script": "..."}\n'
    "- get_current_time: {}\n"
    "Do not explain. Do not use natural language outside the step lines. "
    "The last step should usually be SAY: to summarize results."
)


def plan_and_execute(server, task: str, max_tokens: int, voice_mode: bool = False) -> str | None:
    """Generate a step plan and execute it using local tools."""
    # 1. Ask the 8B for a dry, structured plan.
    plan_messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    plan_prompt = server.tokenizer.apply_chat_template(
        plan_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    plan_raw = server._stream(plan_prompt, max_tokens=200, voice_mode=voice_mode)
    plan_lines = [line.strip() for line in plan_raw.splitlines() if line.strip().startswith(("TOOL:", "SAY:"))]
    if not plan_lines:
        return None

    # 2. Execute tool steps, collecting the last tool result.
    last_tool_result = ""
    final_say = ""
    for line in plan_lines:
        if line.startswith("TOOL:"):
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            tool_name = parts[1].strip()
            try:
                args = json.loads(parts[2].strip())
            except json.JSONDecodeError:
                continue
            last_tool_result = server._run_approved_tool(tool_name, args, task)
        elif line.startswith("SAY:"):
            final_say = line.split(":", 1)[1].strip()

    # 3. If a SAY step exists, use it as a prompt to summarize the last tool result.
    if final_say:
        summary_messages = [
            {"role": "system", "content": server.system_prompt},
            {"role": "user", "content": task},
            {"role": "tool", "content": f"Tool result:\n{last_tool_result}"},
            {"role": "user", "content": final_say},
        ]
        summary_prompt = server.tokenizer.apply_chat_template(
            summary_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        raw = server._stream(summary_prompt.rstrip(), max_tokens, voice_mode=voice_mode)
        return postprocess_output(raw.strip())

    # No SAY step: just return the last tool result with persona polish.
    return postprocess_output(last_tool_result)




def extract_agent_json(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a model response for the agent loop."""
    # Try a fenced JSON block first.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Fall back to the first bare JSON object.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None



def extract_agent_xml(text: str) -> dict[str, Any] | None:
    """Convert a Qwen-style <tool_call> into an agent decision."""
    calls, _ = extract_tool_calls(text)
    if not calls:
        return None
    call = calls[0]
    return {
        "thought": text.strip(),
        "tool": call.get("name", ""),
        "args": call.get("arguments") or call.get("args") or {},
    }




def run_agent_task(
    server,
    goal: str,
    max_steps: int = 10,
    voice_mode: bool = False,
    task_id: str | None = None,
) -> str:
    max_steps = max(1, min(int(max_steps), 50))
    """Autonomous plan/act/observe loop for multi-step tasks.

    If task_id is provided, the AgentTaskManager record is updated in place.
    Otherwise a new task is created and tracked.
    """
    max_steps = max(1, min(max_steps, 50))
    task = (
        server.agent_task_manager.get(task_id)
        if task_id
        else server.agent_task_manager.create(goal, max_steps)
    )
    if task is None:
        return f"Agent task {task_id} not found."
    task_id = task.task_id
    server.agent_task_manager.update_status(task_id, "running")

    def _record_step(thought: str, tool: str, args: dict[str, Any], result: str, error: str = "") -> None:
        step = badapple_agent_tasks.AgentStep(
            thought=thought,
            tool=tool,
            args=args,
            result=result[:500],
            error=error,
        )
        server.agent_task_manager.add_step(task_id, step)

    # Pick a focused tool set for the goal so the prompt stays small.
    agent_tools = tools_for_prompt(goal)
    agent_tools = [t for t in agent_tools if t["function"]["name"] != "run_agent_task"]
    if not agent_tools:
        agent_tools = [t for t in TOOLS if t["function"]["name"] != "run_agent_task"][:12]
    tool_names = ", ".join(t["function"]["name"] for t in agent_tools)
    tool_docs = "\n".join(
        f"- {t['function']['name']}: {t['function'].get('description', '')}\n  args: {json.dumps(t['function'].get('parameters', {}))}"
        for t in agent_tools
    )

    history_for_model: list[dict[str, Any]] = []
    system_prompt = (
        "Bad Apple is a local AI operating system layer for macOS. "
        "The language model is an internal component, not Bad Apple's identity; Bad Apple is not merely a text LLM or AI wrapper. "
        "You are an autonomous agent inside that operating system. "
        "You have a goal and a focused set of tools. "
        "Think step by step. For each step, output a single JSON object with one of these shapes:\n"
        '1. To take an action: {"thought": "...", "tool": "tool_name", "args": {...}}\n'
        '2. To finish the task: {"thought": "...", "finish": "final answer to the user"}\n\n'
        "Important: 'finish' is NOT a tool. When the task is done, emit the finish JSON and do not call any tool.\n"
        "Available tools: " + tool_names + "\n\n"
        + tool_docs + "\n\n"
        "Rules:\n"
        "- Output ONLY the JSON object. No markdown, no explanation outside the JSON.\n"
        "- Choose the right tool for each step.\n"
        "- If a tool returns an error or unexpected result, decide whether to retry with different arguments, try a different tool, or finish with what you know.\n"
        "- Do not repeat the same failed action more than once without changing something.\n"
        "- Keep going until the goal is fully achieved or you are stuck."
    )

    last_tool_name = ""
    repeated_failures = 0
    for step in range(max_steps):
        if server.agent_task_manager.get(task_id).status == "cancelled":  # type: ignore[union-attr]
            return f"Agent task {task_id} cancelled."

        recent_history = history_for_model[-3:]
        step_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Goal: {goal}\n\nHistory so far:\n{json.dumps(recent_history, indent=2, default=str)}\n\nWhat is the next step?"},
        ]
        prompt_text = server.tokenizer.apply_chat_template(
            step_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        raw = server._stream(prompt_text, 220, voice_mode=voice_mode)

        decision = server._extract_agent_json(raw)
        if decision is None:
            decision = server._extract_agent_xml(raw)

        if decision is None:
            _record_step("", "", {}, raw[:500], "could not parse agent JSON")
            continue

        thought = str(decision.get("thought", ""))
        if "finish" in decision:
            finish = str(decision["finish"])
            server.agent_task_manager.update_status(task_id, "completed", summary=finish)
            return finish

        tool_name = decision.get("tool", "")
        tool_args = decision.get("args", {})
        if tool_name == "finish":
            finish = str(decision.get("finish", decision.get("args", json.dumps(decision))))
            server.agent_task_manager.update_status(task_id, "completed", summary=finish)
            return finish
        if not tool_name:
            _record_step(thought, "", {}, raw[:500], "no tool chosen")
            continue

        result = server._run_approved_tool(tool_name, tool_args, f"agent task: {goal}")
        _record_step(thought, tool_name, tool_args, str(result))
        history_for_model.append({
            "step": step,
            "thought": thought,
            "tool": tool_name,
            "args": tool_args,
            "result": str(result)[:280],
        })

        if str(result).startswith(("Approval required", "Policy:", "Runtime", "Tool error")):
            server.agent_task_manager.update_status(task_id, "paused", error=str(result))
            return f"Agent task {task_id} paused: {result}"

        if tool_name == last_tool_name and str(result).startswith("Tool error"):
            repeated_failures += 1
        else:
            repeated_failures = 0
        last_tool_name = tool_name
        if repeated_failures >= 2:
            server.agent_task_manager.update_status(
                task_id, "failed", error="Repeated failures on the same tool."
            )
            return f"Agent task {task_id} failed after repeated errors."

    progress = [s.to_dict() for s in server.agent_task_manager.get(task_id).steps]  # type: ignore[union-attr]
    server.agent_task_manager.update_status(
        task_id, "failed", error=f"Reached step limit ({max_steps})."
    )
    return (
        f'Agent task {task_id} for "{goal}" reached the step limit '
        f"({max_steps}).\n\nProgress:\n{json.dumps(progress, indent=2, default=str)}"
    )




def agent_done(server, future: Any, task_id: str) -> None:
    try:
        future.result()
    except Exception as e:  # noqa: BLE001
        server.agent_task_manager.update_status(
            task_id, "failed", error=f"Uncaught exception: {e}"
        )




def submit_agent_task(server, goal: str, max_steps: int = 10) -> badapple_agent_tasks.AgentTask:
    """Queue a background agent task and return immediately."""
    task = server.agent_task_manager.create(goal, max_steps)
    if not getattr(server, "loop", None) or not getattr(server, "executor", None):
        return task
    future = server.loop.run_in_executor(
        server.executor,
        server.run_agent_task,
        goal,
        max_steps,
        False,
        task.task_id,
    )
    future.add_done_callback(lambda fut: server._agent_done(fut, task.task_id))
    return task




def list_agent_tasks(server) -> list[dict[str, Any]]:
    return server.agent_task_manager.status()




def get_agent_task(server, task_id: str) -> dict[str, Any] | None:
    task = server.agent_task_manager.get(task_id)
    if task is None:
        return None
    return task.to_dict()




def cancel_agent_task(server, task_id: str) -> bool:
    return server.agent_task_manager.cancel(task_id)




def pause_agent_task(server, task_id: str) -> bool:
    return server.agent_task_manager.pause(task_id)




def resume_agent_task(server, task_id: str) -> bool:
    ok = server.agent_task_manager.resume(task_id)
    if not ok:
        return False
    task = server.agent_task_manager.get(task_id)
    if task is None:
        return False
    if not getattr(server, "loop", None) or not getattr(server, "executor", None):
        return True
    future = server.loop.run_in_executor(
        server.executor,
        server.run_agent_task,
        task.goal,
        task.max_steps,
        False,
        task.task_id,
    )
    future.add_done_callback(lambda fut: server._agent_done(fut, task.task_id))
    return True




async def handle_agent_request(server, raw: str, writer: asyncio.StreamWriter):
    """Minimal local agent protocol (LAP) over SLICKS.

    Request envelope (JSON, embedded after the `__BADAPPLE_AGENT__ ` sentinel):
        {"id": "req-1", "method": "discover_tools"}
        {"id": "req-2", "method": "invoke_tool", "params": {"name": "run_shell", "args": {"command": "ls"}}}
        {"id": "req-3", "method": "inference", "params": {"prompt": "what is 2+2?", "max_new_tokens": 120}}
    """

    async def _respond(req_id: str | None, result: Any, error: str | None = None):
        frame: dict[str, Any] = {"id": req_id}
        if error:
            frame["type"] = "error"
            frame["message"] = error
        else:
            frame["type"] = "response"
            frame["result"] = result
        await _write_frame(writer, frame)

    try:
        req = json.loads(raw[len("__BADAPPLE_AGENT__ "):])
    except json.JSONDecodeError as e:
        await _respond(None, None, f"invalid agent JSON: {e}")
        return

    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params") or {}

    if not server._is_passive_method(method):
        server.touch_activity()

    if method == "set_hibernate_after":
        seconds = float(params.get("seconds", 300))
        server.hibernate_after = max(0, seconds)
        await _respond(req_id, {"hibernate_after": server.hibernate_after})
        return

    if method == "model_status":
        await _respond(req_id, server.model_manager.status(params.get("model_id")))
        return

    if method == "list_models":
        await _respond(req_id, {"text": server.model_registry.list_models(), "models": server.model_registry._state.get("models", [])})
        return

    if method == "scan_models":
        await _respond(req_id, {"text": server.model_registry.scan(), "models": server.model_registry._state.get("models", [])})
        return

    if method == "model_info":
        model_id = str(params.get("model_id", ""))
        if not model_id:
            await _respond(req_id, None, "model_id is required")
            return
        await _respond(req_id, {"text": server.model_registry.info(model_id)})
        return

    if method == "verify_models":
        model_id = params.get("model_id")
        result = server.model_registry.verify(model_id)
        await _respond(req_id, result)
        return

    if method == "add_model":
        path = str(params.get("path", ""))
        model_id = str(params.get("model_id", ""))
        if not path:
            await _respond(req_id, None, "path is required")
            return
        result = server.model_registry.add_model(path, model_id)
        await _respond(req_id, result)
        return

    if method == "remove_model":
        model_id = str(params.get("model_id", ""))
        if not model_id:
            await _respond(req_id, None, "model_id is required")
            return
        result = server.model_registry.remove_model(model_id)
        await _respond(req_id, result)
        return

    if method == "download_model":
        model_id = str(params.get("model_id", ""))
        if not model_id:
            await _respond(req_id, None, "model_id is required")
            return
        if not server.model_manager.allow_downloads:
            await _respond(req_id, None, "downloads disabled; call set_allow_downloads first")
            return
        await _respond(req_id, server.model_manager.start_download(model_id))
        return

    if method == "set_allow_downloads":
        enabled = bool(params.get("enabled", False))
        if server.airgap and enabled:
            await _respond(req_id, None, "downloads cannot be enabled while air-gap mode is on")
            return
        server.model_manager.set_allow_downloads(enabled)
        await _respond(req_id, {"allow_downloads": enabled})
        return

    if method == "set_airgap":
        enabled = bool(params.get("enabled", False))
        server._set_airgap(enabled)
        await _respond(req_id, {"airgap": server.airgap})
        return
    if method == "airgap_status":
        await _respond(req_id, {"airgap": server.airgap})
        return

    if method == "recommend_model":
        await _respond(req_id, server.recommend_model(str(params.get("query", ""))))
        return

    if method == "admit_model":
        model_ref = str(params.get("model_ref", ""))
        if not model_ref:
            await _respond(req_id, None, "model_ref is required")
            return
        await _respond(req_id, server.admit_model(model_ref, bool(params.get("auto_unload", True))))
        return

    if method == "switch_main_model":
        model_ref = str(params.get("model_ref", ""))
        if not model_ref:
            await _respond(req_id, None, "model_ref is required")
            return
        # Run the heavy load in the MLX executor.
        if not getattr(server, "loop", None) or not getattr(server, "executor", None):
            await _respond(req_id, None, "server not initialized")
            return
        future = server.loop.run_in_executor(server.executor, server.switch_main_model, model_ref)
        result = await future
        await _respond(req_id, {"result": result})
        return

    if method == "preload_models":
        model_ids = params.get("model_ids")
        if isinstance(model_ids, str):
            model_ids = [m.strip() for m in model_ids.split(",") if m.strip()]
        result = server.submit_preload_models(model_ids)
        await _respond(req_id, {"preloaded": result})
        return

    if method == "run_agent_task":
        goal = str(params.get("goal", ""))
        max_steps = int(params.get("max_steps") or 10)
        if not goal:
            await _respond(req_id, None, "goal is required")
            return
        task = server.submit_agent_task(goal, max_steps)
        await _respond(req_id, {"task_id": task.task_id, "status": task.status, "goal": task.goal})
        return

    if method == "list_agent_tasks":
        await _respond(req_id, {"tasks": server.list_agent_tasks()})
        return

    if method == "get_agent_task":
        task = server.get_agent_task(str(params.get("task_id", "")))
        await _respond(req_id, {"task": task})
        return

    if method == "cancel_agent_task":
        ok = server.cancel_agent_task(str(params.get("task_id", "")))
        await _respond(req_id, {"cancelled": ok})
        return

    if method == "pause_agent_task":
        ok = server.pause_agent_task(str(params.get("task_id", "")))
        await _respond(req_id, {"paused": ok})
        return

    if method == "resume_agent_task":
        ok = server.resume_agent_task(str(params.get("task_id", "")))
        await _respond(req_id, {"resumed": ok})
        return

    if method == "runtime_status":
        ambient = None
        try:
            ambient = json.loads(badapple_ambient.get_context()) if badapple_ambient._CONTEXT_FILE.is_file() else None
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError) as e:
            print(f"[mlx_server] is_file failed: {e}", flush=True)
        ambient_running = badapple_ambient.is_running()
        ocular = None
        try:
            ocular = badapple_ocular.status() if badapple_ocular.OCULAR_CONTEXT.is_file() else None
        except Exception as e:  # noqa: BLE001
            print(f"[mlx_server] ocular status error: {e}", flush=True)
        await _respond(req_id, {
            "runtime": server.runtime.status(),
            "health": server.health.snapshot(),
            "resources": server.resources.snapshot(),
            "active_models": server.active_models(),
            "breakers": server.breakers.snapshot_all(),
            "autopilot": server.policy.autopilot,
            "fast_tier": server.fast_tier_enabled,
            "ambient_running": ambient_running,
            "ambient": ambient,
            "ocular_running": badapple_ocular.is_running(),
            "ocular": ocular,
            "workspace": str(server.workspace.path) if server.workspace.path else None,
            "airgap": server.airgap,
            "p2p_enabled": server.p2p is not None and server.p2p.is_running(),
            "p2p_peers": server.p2p.get_peers() if server.p2p is not None and server.p2p.is_running() else [],
            "mcp_socket": os.environ.get("BADAPPLE_MCP_SOCKET", "/var/run/badapple/mcp.sock"),
            "fast_model": server.fast_model_info,
            "main_model_loaded": server.model is not None and server.tokenizer is not None,
            "models": server.model_manager.status() if getattr(server, "model_manager", None) is not None else {},
            "agent_tasks": server.list_agent_tasks(),
            "hibernating": server.hibernating,
            "idle_seconds": round(time.time() - server.last_activity, 1),
        })
        return

    if method == "identity_status":
        await _respond(req_id, {"status": badapple_identity.status(), "public_key": badapple_identity.public_key()})
        return

    if method == "identity_sign":
        challenge = str(params.get("challenge", ""))
        if not challenge or len(challenge) > 4096:
            await _respond(req_id, None, "identity_sign requires a challenge up to 4096 characters")
            return
        await _respond(req_id, {"signature": badapple_identity.sign(challenge.encode("utf-8"))})
        return

    if method == "kill_switch":
        enabled = bool(params.get("enabled", True))
        if enabled:
            state = server.runtime.engage_kill_switch(params.get("reason", "agent requested"))
        else:
            state = server.runtime.reset_kill_switch()
            if state.get("safe_mode_reason"):
                state = server.runtime.leave_safe_mode()
        await _respond(req_id, {"runtime": state})
        return

    if method == "private_mode":
        state = server.runtime.set_private_mode(bool(params.get("enabled", True)))
        await _respond(req_id, {"runtime": state})
        return

    if method == "discover_tools":
        await _respond(req_id, {"tools": TOOLS})
        return

    if method == "invoke_tool":
        if not server.runtime.allows_mutation():
            await _respond(req_id, None, "runtime is stopped or in safe mode")
            return
        tool_name = params.get("name", "")
        tool_args = params.get("args") or {}
        result = server._run_approved_tool(tool_name, tool_args, "agent request")
        await _respond(req_id, {"tool": tool_name, "result": result})
        return

    if method == "set_fast_tier":
        server.fast_tier_enabled = bool(params.get("enabled", True))
        await _respond(req_id, {"fast_tier": server.fast_tier_enabled})
        return

    if method == "set_autopilot":
        server.policy.set_autopilot(bool(params.get("enabled", False)))
        await _respond(req_id, {"autopilot": server.policy.autopilot})
        return

    if method == "inference":
        if not server.runtime.allows_generation():
            await _respond(req_id, None, "kill switch is engaged")
            return
        prompt = params.get("prompt", "")
        max_tokens = int(params.get("max_new_tokens", 120))
        if not prompt:
            await _respond(req_id, None, "inference requires prompt")
            return
        loop = asyncio.get_event_loop()

        def _gen():
            mx.set_default_device(server.mlx_device)
            try:
                # The inference API is stateless: it must not mutate the
                # conversational turn cache or return a cached conversational
                # response. Build a single-turn prompt and stream directly.
                # Ensure the lazily-loaded main model (and its tokenizer) exist
                # before render_prompt needs them.
                server._ensure_main_model()
                messages = [
                    {"role": "system", "content": server.personas.get_system_prompt()},
                    {"role": "user", "content": prompt},
                ]
                rendered = server.render_prompt(messages, use_tools=False, voice_mode=False)
                raw = server._stream(rendered, max_tokens, voice_mode=False)
                text = server.polish_response(raw)
                server._audit_record("query", {
                    "prompt": prompt,
                    "persona": server.personas.active,
                    "voice_mode": False,
                    "use_tools": False,
                })
                server._audit_record("response", {
                    "prompt": prompt,
                    "response": text[:500],
                    "persona": server.personas.active,
                })
                return text
            except Exception as e:  # noqa: BLE001 - catch-all wrapper
                traceback.print_exc()
                return f"Error generating response: {e}"

        text = await loop.run_in_executor(server.executor, _gen)
        metrics = server.last_metrics
        await _respond(req_id, {"text": text, "metrics": metrics})
        return

    if method == "switch_persona":
        name = params.get("name", "")
        if server.personas.switch(name):
            await _respond(req_id, {"active_persona": server.personas.active})
        else:
            await _respond(req_id, None, f"unknown persona '{name}'")
        return

    if method == "set_workspace":
        path = params.get("path", "")
        if not path:
            await _respond(req_id, None, "set_workspace requires path")
            return
        result = server.workspace.set(path)
        await _respond(req_id, {"status": result})
        return

    if method == "get_workspace":
        summary = server.workspace.summary() if server.workspace.path else None
        await _respond(req_id, {"workspace": str(server.workspace.path) if server.workspace.path else None, "summary": summary})
        return

    if method == "set_p2p":
        enabled = bool(params.get("enabled", False))
        if server.p2p is None:
            await _respond(req_id, None, "P2P is not available")
            return
        try:
            if enabled:
                await asyncio.to_thread(server.p2p.start)
            else:
                await asyncio.to_thread(server.p2p.stop)
        except Exception as e:  # noqa: BLE001 - catch-all wrapper
            await _respond(req_id, None, f"P2P toggle failed: {e}")
            return
        await _respond(req_id, {"p2p_enabled": server.p2p.is_running()})
        return

    if method == "set_ambient":
        enabled = bool(params.get("enabled", False))
        text = badapple_ambient.start() if enabled else badapple_ambient.stop()
        await _respond(req_id, {"ambient_running": badapple_ambient.is_running(), "message": text})
        return

    if method == "set_ocular":
        enabled = bool(params.get("enabled", False))
        text = badapple_ocular.start(
            float(params.get("capture_interval", 5)),
            float(params.get("describe_interval", 0)),
            params.get("prompt"),
        ) if enabled else badapple_ocular.stop()
        await _respond(req_id, {"ocular_running": badapple_ocular.is_running(), "message": text})
        return

    if method == "audit_tail":
        n = int(params.get("n", 20))
        entries = []
        if server.audit_collector.ledger_path.is_file():
            try:
                with open(server.audit_collector.ledger_path, encoding="utf-8") as f:
                    lines = f.readlines()
                entries = [json.loads(line) for line in lines[-n:] if line.strip()]
            except (json.JSONDecodeError, TypeError, ValueError, AttributeError, OSError) as e:
                await _respond(req_id, None, f"could not read ledger: {e}")
                return
        await _respond(req_id, {"entries": entries})
        return

    if method == "flush_vram":
        result = server.flush_vram()
        await _respond(req_id, {"result": result})
        return

    if method == "unload_model":
        model_type = str(params.get("type", "vision"))
        result = server.unload_model(model_type)
        await _respond(req_id, {"result": result})
        return

    if method == "get_pending_approvals":
        await _respond(req_id, {"pending": server.approval.get_pending_summary()})
        return
    if method == "audit_checkpoint":
        # Actor .ask() blocks on a queue; run off the event loop thread.
        result = await asyncio.get_event_loop().run_in_executor(
            server.executor, server.audit_actor.ask, {"method": "sign_checkpoint"}
        )
        await _respond(req_id, {"result": result})
        return
    if method == "audit_verify":
        results = await asyncio.get_event_loop().run_in_executor(
            server.executor, server.audit_actor.ask, {"method": "verify"}
        )
        invalid = [r for r in (results or []) if not r.get("valid")]
        await _respond(req_id, {"result": {
            "total_entries": len(results or []),
            "invalid_entries": len(invalid),
            "valid": not invalid,
            "first_invalid": invalid[0] if invalid else None,
        }})
        return
    if method == "p2p_peers":
        daemon = badapple_p2p.get_p2p_daemon()
        if daemon is None:
            await _respond(req_id, None, "P2P is off. Turn it on from the menu bar or with `badapple p2p on`.")
            return
        await _respond(req_id, {"peers_summary": daemon.get_peers()})
        return
    if method == "p2p_sync":
        daemon = badapple_p2p.get_p2p_daemon()
        if daemon is None:
            await _respond(req_id, None, "P2P is off. Turn it on from the menu bar or with `badapple p2p on`.")
            return
        try:
            result = await asyncio.to_thread(daemon.sync_memory)
        except Exception as e:  # noqa: BLE001 - catch-all wrapper
            await _respond(req_id, None, f"P2P sync failed: {e}")
            return
        await _respond(req_id, {"sync_status": result})
        return
    if method == "p2p_models":
        if server.p2p is None:
            await _respond(req_id, None, "P2P is off. Turn it on from the menu bar or with `badapple p2p on`.")
            return
        try:
            result = await asyncio.to_thread(server.p2p.remote_models)
        except Exception as e:  # noqa: BLE001 - catch-all wrapper
            await _respond(req_id, None, f"P2P models failed: {e}")
            return
        await _respond(req_id, result)
        return
    if method == "p2p_pull_model":
        if server.p2p is None:
            await _respond(req_id, None, "P2P is off. Turn it on from the menu bar or with `badapple p2p on`.")
            return
        peer_id = str(params.get("peer_id", ""))
        model_id = str(params.get("model_id", ""))
        if not peer_id or not model_id:
            await _respond(req_id, None, "Both a peer and a model name are needed to pull a model.")
            return
        try:
            result = await asyncio.to_thread(server.p2p.pull_model_manifest, peer_id, model_id)
        except Exception as e:  # noqa: BLE001 - catch-all wrapper
            await _respond(req_id, None, f"P2P pull failed: {e}")
            return
        await _respond(req_id, result)
        return
    if method == "p2p_send_model":
        if server.p2p is None:
            await _respond(req_id, None, "P2P is off. Turn it on from the menu bar or with `badapple p2p on`.")
            return
        peer_id = str(params.get("peer_id", ""))
        model_id = str(params.get("model_id", ""))
        if not peer_id or not model_id:
            await _respond(req_id, None, "Both a peer and a model name are needed to send a model.")
            return
        try:
            result = await asyncio.to_thread(server.p2p.send_model, peer_id, model_id)
        except Exception as e:  # noqa: BLE001 - catch-all wrapper
            await _respond(req_id, None, f"P2P send failed: {e}")
            return
        await _respond(req_id, {"send_status": result})
        return
    if method == "p2p_receive_model":
        if server.p2p is None:
            await _respond(req_id, None, "P2P is off. Turn it on from the menu bar or with `badapple p2p on`.")
            return
        peer_id = str(params.get("peer_id", ""))
        model_id = str(params.get("model_id", ""))
        try:
            result = await asyncio.to_thread(server.p2p.receive_model, peer_id, model_id)
        except Exception as e:  # noqa: BLE001 - catch-all wrapper
            await _respond(req_id, None, f"P2P receive failed: {e}")
            return
        await _respond(req_id, result)
        return

    await _respond(req_id, None, f"unknown method '{method}'")

