#!/usr/bin/env python3
"""Run selected OpenThoughts-TBLite tasks through Hermes AIAgent.

This differs from tblite_env.py: it instantiates the real Hermes ``AIAgent``
using the current Hermes config, while still using TBLite's Docker task images
and verifier scripts.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import shlex
import shutil
import signal
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HERMES_HOME = Path.home() / ".hermes"
DEFAULT_TASKS = [
    "amuse-install",
    "anomaly-detection-ranking",
    "basic-message-queue",
    "book-portfolio-analysis",
    "broken-python",
]


def _json_default(value: Any) -> str:
    return str(value)


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _copy_if_exists(src: Path, dest: Path) -> None:
    if src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _prepare_isolated_home(source_home: Path, output_dir: Path) -> Path:
    hermes_home = output_dir / "hermes-home"
    hermes_home.mkdir(parents=True, exist_ok=True)
    for dirname in ("logs", "sessions", "skills"):
        (hermes_home / dirname).mkdir(parents=True, exist_ok=True)

    _copy_if_exists(source_home / "config.yaml", hermes_home / "config.yaml")
    _copy_if_exists(source_home / ".env", hermes_home / ".env")
    _copy_if_exists(source_home / "SOUL.md", hermes_home / "SOUL.md")
    return hermes_home


def _patch_isolated_terminal_config(
    hermes_home: Path,
    *,
    backend: str,
    timeout: int,
    lifetime_seconds: int,
    max_foreground_timeout: int,
) -> None:
    """Make the copied config authoritative for the benchmark terminal backend.

    Importing ``cli`` bridges ``terminal.*`` config values into TERMINAL_* env
    vars. If the copied config says ``local``, it overrides the process env.
    Patch the isolated copy instead of touching the user's real config.
    """
    import yaml

    config_path = hermes_home / "config.yaml"
    payload: Dict[str, Any] = {}
    if config_path.exists():
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    terminal = payload.setdefault("terminal", {})
    terminal["backend"] = backend
    terminal["timeout"] = timeout
    terminal["lifetime_seconds"] = lifetime_seconds
    terminal["container_persistent"] = True
    terminal["docker_mount_cwd_to_workspace"] = False
    terminal.pop("cwd", None)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    # The copied user .env is loaded with override=True by run_agent.py.  Keep
    # its terminal values in sync with the isolated benchmark config so later
    # dotenv reloads cannot revert the sandbox lifetime back to the user's
    # interactive defaults.
    env_path = hermes_home / ".env"
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    terminal_env = {
        "TERMINAL_ENV": backend,
        "TERMINAL_TIMEOUT": str(timeout),
        "TERMINAL_LIFETIME_SECONDS": str(lifetime_seconds),
        "TERMINAL_MAX_FOREGROUND_TIMEOUT": str(max_foreground_timeout),
        "TERMINAL_CONTAINER_PERSISTENT": "true",
        "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE": "false",
    }
    lines = []
    seen = set()
    for line in existing.splitlines():
        key = line.split("=", 1)[0].strip()
        if key in terminal_env:
            lines.append(f"{key}={terminal_env[key]}")
            seen.add(key)
        else:
            lines.append(line)
    for key, value in terminal_env.items():
        if key not in seen:
            lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _safe_extract_tar(tar: tarfile.TarFile, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    for member in tar.getmembers():
        name = member.name.replace("\\", "/")
        parts = [p for p in name.split("/") if p not in ("", ".")]
        if not parts or any(p == ".." for p in parts) or name.startswith("/"):
            raise ValueError(f"Unsafe archive member path: {member.name}")
        target = target_dir.joinpath(*parts)
        target_real = target.resolve(strict=False)
        try:
            target_real.relative_to(target_root)
        except ValueError as exc:
            raise ValueError(f"Unsafe archive member path: {member.name}") from exc

        if member.isdir():
            target_real.mkdir(parents=True, exist_ok=True)
            continue
        if not member.isfile():
            raise ValueError(f"Unsupported archive member type: {member.name}")

        target_real.parent.mkdir(parents=True, exist_ok=True)
        extracted = tar.extractfile(member)
        if extracted is None:
            raise ValueError(f"Cannot read archive member: {member.name}")
        with extracted, open(target_real, "wb") as dst:
            shutil.copyfileobj(extracted, dst)


def _extract_base64_tar(b64_data: str, target_dir: Path) -> None:
    if not b64_data:
        return
    raw = base64.b64decode(b64_data)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        _safe_extract_tar(tar, target_dir)


class _VerifierToolContext:
    """Small self-contained tool context for benchmark verification.

    Upstream Hermes removed the old Atropos ``environments.tool_context``
    package. The TBLite smoke still needs verifier access to the same task
    sandbox the actor used, so keep the few required calls local to this
    benchmark runner.
    """

    def __init__(self, task_id: str):
        self.task_id = task_id

    def _call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        from model_tools import handle_function_call

        raw = handle_function_call(name, args, task_id=self.task_id)
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"result": parsed}
        except Exception:
            return {"exit_code": -1, "output": str(raw)}

    def terminal(self, command: str, timeout: int = 180) -> Dict[str, Any]:
        return self._call_tool("terminal", {"command": command, "timeout": timeout})

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        return self._call_tool("write_file", {"path": path, "content": content})

    def upload_dir(self, local_dir: str, remote_dir: str) -> Dict[str, Any]:
        local = Path(local_dir)
        if not local.is_dir():
            return {"exit_code": -1, "output": f"Local directory not found: {local_dir}"}

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for path in sorted(local.rglob("*")):
                if path.is_file():
                    tar.add(path, arcname=str(path.relative_to(local)))

        archive_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        remote_b64 = f"/tmp/hermes_verifier_upload_{uuid.uuid4().hex}.tar.gz.b64"
        remote_tar = remote_b64[:-4]
        chunk_size = 60_000
        self.terminal(f"mkdir -p {shlex.quote(remote_dir)} && : > {shlex.quote(remote_b64)}", timeout=30)
        for idx in range(0, len(archive_b64), chunk_size):
            chunk = archive_b64[idx : idx + chunk_size]
            result = self.terminal(
                f"printf %s {shlex.quote(chunk)} >> {shlex.quote(remote_b64)}",
                timeout=30,
            )
            if int(result.get("exit_code", -1)) != 0:
                return result
        return self.terminal(
            "base64 -d {b64} > {tar} && tar -xzf {tar} -C {dest} && rm -f {b64} {tar}".format(
                b64=shlex.quote(remote_b64),
                tar=shlex.quote(remote_tar),
                dest=shlex.quote(remote_dir),
            ),
            timeout=120,
        )

    def download_dir(self, remote_dir: str, local_dir: str) -> Dict[str, Any]:
        remote_tar = f"/tmp/hermes_verifier_download_{uuid.uuid4().hex}.tar.gz"
        result = self.terminal(
            "cd {src} && tar -czf {tar} . && base64 -w 0 {tar} && rm -f {tar}".format(
                src=shlex.quote(remote_dir),
                tar=shlex.quote(remote_tar),
            ),
            timeout=120,
        )
        if int(result.get("exit_code", -1)) != 0:
            return result

        encoded = "".join(str(result.get("output", "")).split())
        try:
            raw = base64.b64decode(encoded)
        except Exception as exc:
            return {"success": False, "error": f"failed to decode verifier archive: {exc}"}

        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
                _safe_extract_tar(tar, target)
        except Exception as exc:
            return {"success": False, "error": f"failed to extract verifier archive: {exc}"}
        return {"success": True, "bytes": len(raw)}


class _AlarmTimeout:
    def __init__(self, seconds: int):
        self.seconds = seconds
        self.previous_handler = None

    def __enter__(self):
        if self.seconds <= 0:
            return self
        self.previous_handler = signal.getsignal(signal.SIGALRM)

        def _raise_timeout(signum, frame):
            del signum, frame
            raise TimeoutError(f"task exceeded {self.seconds}s")

        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.previous_handler)
        return False


def _load_selected_tasks(task_names: List[str], dataset_name: str, split: str) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    wanted = set(task_names)
    tasks = [dict(row) for row in load_dataset(dataset_name, split=split) if row["task_name"] in wanted]
    by_name = {task["task_name"]: task for task in tasks}
    missing = [name for name in task_names if name not in by_name]
    if missing:
        raise RuntimeError(f"Missing task(s) in dataset: {missing}")
    return [by_name[name] for name in task_names]


def _run_tests(item: Dict[str, Any], task_id: str, test_timeout: int) -> Dict[str, Any]:
    ctx = _VerifierToolContext(task_id)
    task_name = item.get("task_name", "unknown")
    tests_tar = item.get("tests_tar", "")
    test_sh = item.get("test_sh", "")
    if not test_sh:
        return {"reward": 0.0, "error": "missing test_sh"}

    ctx.terminal("mkdir -p /tests /logs/verifier", timeout=30)
    if tests_tar:
        tests_temp = Path(tempfile.mkdtemp(prefix=f"tblite-tests-{task_name}-"))
        try:
            _extract_base64_tar(tests_tar, tests_temp)
            ctx.upload_dir(str(tests_temp), "/tests")
        finally:
            shutil.rmtree(tests_temp, ignore_errors=True)

    ctx.write_file("/tests/test.sh", test_sh)
    ctx.terminal("chmod +x /tests/test.sh", timeout=30)
    test_result = ctx.terminal("bash /tests/test.sh", timeout=test_timeout)

    reward = 0.0
    local_verifier = Path(tempfile.mkdtemp(prefix=f"tblite-verifier-{task_name}-"))
    try:
        ctx.download_dir("/logs/verifier", str(local_verifier))
        reward_file = local_verifier / "reward.txt"
        if reward_file.exists() and reward_file.stat().st_size > 0:
            content = reward_file.read_text(encoding="utf-8", errors="replace").strip()
            if content == "1":
                reward = 1.0
            elif content == "0":
                reward = 0.0
            else:
                try:
                    reward = float(content)
                except ValueError:
                    reward = 1.0 if int(test_result.get("exit_code", -1)) == 0 else 0.0
        else:
            reward = 1.0 if int(test_result.get("exit_code", -1)) == 0 else 0.0
    finally:
        shutil.rmtree(local_verifier, ignore_errors=True)

    return {
        "reward": reward,
        "test_exit_code": test_result.get("exit_code"),
        "test_output_tail": str(test_result.get("output", ""))[-4000:],
    }


def _summarize_turns(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    tool_counts: Dict[str, int] = {}
    assistant_turns = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        assistant_turns += 1
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            name = (tc.get("function") or {}).get("name") or tc.get("name") or ""
            if name:
                tool_counts[name] = tool_counts.get(name, 0) + 1
    return {"assistant_turns": assistant_turns, "tool_call_counts": tool_counts}


def _build_agent(config: Dict[str, Any], *, max_turns: int, max_tokens: Optional[int], toolsets: List[str]):
    from cli import _parse_reasoning_config
    from hermes_state import SessionDB
    from run_agent import AIAgent

    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    agent_cfg = config.get("agent", {}) if isinstance(config.get("agent"), dict) else {}
    checkpoints_cfg = config.get("checkpoints", {})
    if isinstance(checkpoints_cfg, bool):
        checkpoints_cfg = {"enabled": checkpoints_cfg}
    elif not isinstance(checkpoints_cfg, dict):
        checkpoints_cfg = {}

    effective_max_tokens = max_tokens
    if effective_max_tokens is None and model_cfg.get("max_tokens") is not None:
        effective_max_tokens = int(model_cfg["max_tokens"])

    return AIAgent(
        model=str(model_cfg.get("default") or ""),
        api_key=str(model_cfg.get("api_key") or ""),
        base_url=str(model_cfg.get("base_url") or ""),
        provider=str(model_cfg.get("provider") or ""),
        api_mode=str(model_cfg.get("api_mode") or ""),
        responses_stateful=bool(model_cfg.get("responses_stateful", False)),
        max_iterations=max_turns,
        enabled_toolsets=toolsets,
        save_trajectories=False,
        verbose_logging=False,
        quiet_mode=True,
        ephemeral_system_prompt=str(agent_cfg.get("system_prompt") or "") or None,
        reasoning_config=_parse_reasoning_config(str(agent_cfg.get("reasoning_effort") or "")),
        platform="cli",
        skip_context_files=True,
        skip_memory=True,
        session_db=SessionDB(),
        checkpoints_enabled=bool(checkpoints_cfg.get("enabled", False)),
        checkpoint_max_snapshots=int(checkpoints_cfg.get("max_snapshots", 50)),
        max_tokens=effective_max_tokens,
    )


def _task_image(item: Dict[str, Any]) -> str:
    image = str(item.get("docker_image") or "").strip()
    if not image:
        raise RuntimeError(f"Task {item.get('task_name')} has no docker_image")
    return image


def run_task(
    item: Dict[str, Any],
    config: Dict[str, Any],
    *,
    max_turns: int,
    max_tokens: Optional[int],
    task_timeout: int,
    test_timeout: int,
    toolsets: List[str],
) -> Dict[str, Any]:
    from tools.terminal_tool import clear_task_env_overrides, cleanup_vm, register_task_env_overrides

    task_name = item.get("task_name", "unknown")
    category = item.get("category", "unknown")
    task_id = f"tblite_{task_name}_{uuid.uuid4().hex[:8]}"
    started = time.time()
    agent = None

    register_task_env_overrides(task_id, {
        "docker_image": _task_image(item),
        "modal_image": _task_image(item),
        "cwd": "/app",
    })

    try:
        with _AlarmTimeout(task_timeout):
            agent = _build_agent(config, max_turns=max_turns, max_tokens=max_tokens, toolsets=toolsets)
            result = agent.run_conversation(str(item["instruction"]), task_id=task_id)
            verification = _run_tests(item, task_id, test_timeout)
    except BaseException as exc:
        result = {}
        verification = {"reward": 0.0, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if agent is not None:
            try:
                agent.close()
            except Exception:
                pass
        clear_task_env_overrides(task_id)
        try:
            cleanup_vm(task_id)
        except Exception:
            pass

    messages = result.get("messages") or []
    turn_summary = _summarize_turns(messages)
    reward = float(verification.get("reward") or 0.0)
    return {
        "task_name": task_name,
        "category": category,
        "passed": reward == 1.0,
        "reward": reward,
        "elapsed_s": round(time.time() - started, 3),
        "task_id": task_id,
        "agent_completed": bool(result.get("completed")),
        "api_calls": int(result.get("api_calls") or 0),
        "final_response": result.get("final_response"),
        "input_tokens": int(result.get("input_tokens") or 0),
        "output_tokens": int(result.get("output_tokens") or 0),
        "cache_read_tokens": int(result.get("cache_read_tokens") or 0),
        "reasoning_tokens": int(result.get("reasoning_tokens") or 0),
        "assistant_turns": turn_summary["assistant_turns"],
        "tool_call_counts": turn_summary["tool_call_counts"],
        "verification": verification,
        "messages": messages,
        "request_trace": result.get("request_trace") or [],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def _write_trace_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "task_name",
        "task_id",
        "passed",
        "kind",
        "actor",
        "request_id",
        "api_call_index",
        "status",
        "api_duration_s",
        "model_call_s",
        "request_build_s",
        "stream_open_s",
        "stream_time_to_first_event_s",
        "stream_read_s",
        "response_parse_s",
        "duration_s",
        "tool_name",
        "execution_mode",
        "prompt_tokens",
        "completion_tokens",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "approx_input_tokens",
        "request_char_count",
        "model",
        "provider",
        "api_mode",
        "base_url",
        "session_id",
        "turn_id",
        "created_at_s",
    ]
    all_fields = set()
    for row in rows:
        all_fields.update(row.keys())
    fieldnames = [name for name in preferred if name in all_fields]
    fieldnames.extend(sorted(name for name in all_fields if name not in fieldnames))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean: Dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple)):
                    clean[key] = json.dumps(value, ensure_ascii=False, default=_json_default)
                else:
                    clean[key] = value
            writer.writerow(clean)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TBLite tasks through current Hermes AIAgent")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--dataset-name", default="NousResearch/openthoughts-tblite")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--hermes-home", type=Path, default=DEFAULT_HERMES_HOME)
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--task-timeout", type=int, default=1200)
    parser.add_argument("--test-timeout", type=int, default=600)
    parser.add_argument("--terminal-timeout", type=int, default=300)
    parser.add_argument("--toolsets", default="terminal,file")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            REPO_ROOT
            / "environments"
            / "benchmarks"
            / "evals"
            / f"openthoughts-tblite-current-hermes-agent-slice5-{_timestamp()}"
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    isolated_home = _prepare_isolated_home(args.hermes_home.expanduser(), output_dir)
    _patch_isolated_terminal_config(
        isolated_home,
        backend="docker",
        timeout=args.terminal_timeout,
        lifetime_seconds=args.task_timeout + 120,
        max_foreground_timeout=max(args.terminal_timeout, args.test_timeout),
    )
    os.environ["HERMES_HOME"] = str(isolated_home)
    os.environ["TERMINAL_ENV"] = "docker"
    os.environ["TERMINAL_TIMEOUT"] = str(args.terminal_timeout)
    os.environ["TERMINAL_LIFETIME_SECONDS"] = str(args.task_timeout + 120)
    os.environ["TERMINAL_MAX_FOREGROUND_TIMEOUT"] = str(max(args.terminal_timeout, args.test_timeout))
    os.environ["TERMINAL_CONTAINER_PERSISTENT"] = "true"
    os.environ["HERMES_SESSION_SOURCE"] = "tblite-current-hermes-agent"
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from hermes_cli.config import load_config

    config = load_config()
    task_names = [name.strip() for name in args.tasks.split(",") if name.strip()]
    toolsets = [name.strip() for name in args.toolsets.split(",") if name.strip()]
    tasks = _load_selected_tasks(task_names, args.dataset_name, args.dataset_split)

    run_config = {
        "driver": "current_hermes_agent",
        "hermes_home_source": str(args.hermes_home.expanduser()),
        "isolated_hermes_home": str(isolated_home),
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "tasks": task_names,
        "toolsets": toolsets,
        "max_turns": args.max_turns,
        "max_tokens": args.max_tokens or (config.get("model", {}) or {}).get("max_tokens"),
        "task_timeout": args.task_timeout,
        "test_timeout": args.test_timeout,
        "terminal_backend": "docker",
        "terminal_timeout": args.terminal_timeout,
        "model": config.get("model", {}),
        "request_trace_jsonl": str(output_dir / "request_trace.jsonl"),
        "request_trace_csv": str(output_dir / "request_trace.csv"),
    }
    _write_json(output_dir / "run_config.json", run_config)

    samples_path = output_dir / "samples.jsonl"
    if samples_path.exists():
        samples_path.unlink()
    trace_jsonl_path = output_dir / "request_trace.jsonl"
    trace_csv_path = output_dir / "request_trace.csv"
    if trace_jsonl_path.exists():
        trace_jsonl_path.unlink()

    print(f"Output: {output_dir}")
    print(f"Model: {(config.get('model', {}) or {}).get('default')}")
    print(f"API mode: {(config.get('model', {}) or {}).get('api_mode')}")
    print(f"Max tokens: {run_config['max_tokens']}")
    print(f"Tasks: {', '.join(task_names)}")
    print("")

    results: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    started = time.time()
    for index, item in enumerate(tasks, 1):
        name = item["task_name"]
        print(f"[{index}/{len(tasks)}] START {name}", flush=True)
        result = run_task(
            item,
            config,
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            task_timeout=args.task_timeout,
            test_timeout=args.test_timeout,
            toolsets=toolsets,
        )
        results.append(result)
        _append_jsonl(samples_path, result)
        for trace_row in result.get("request_trace") or []:
            enriched = dict(trace_row)
            enriched.setdefault("task_name", result.get("task_name"))
            enriched.setdefault("task_id", result.get("task_id"))
            enriched.setdefault("passed", result.get("passed"))
            enriched.setdefault("task_elapsed_s", result.get("elapsed_s"))
            trace_rows.append(enriched)
            _append_jsonl(trace_jsonl_path, enriched)
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{index}/{len(tasks)}] {status} {name} "
            f"elapsed={result['elapsed_s']:.1f}s api_calls={result['api_calls']} "
            f"output_tokens={result['output_tokens']}",
            flush=True,
        )

    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    metrics = {
        "config_general": {
            "model_name": (config.get("model", {}) or {}).get("default"),
            "total_evaluation_time_seconds": time.time() - started,
            "generation_parameters": {
                "max_tokens": run_config["max_tokens"],
                "temperature": "config/default",
                "max_agent_turns": args.max_turns,
                "terminal_backend": "docker",
                "api_mode": (config.get("model", {}) or {}).get("api_mode"),
                "responses_stateful": (config.get("model", {}) or {}).get("responses_stateful"),
            },
        },
        "results": {
            "all": {
                "eval/pass_rate": passed / total if total else 0.0,
                "eval/total_tasks": total,
                "eval/passed_tasks": passed,
                "eval/evaluation_time_seconds": time.time() - started,
            }
        },
    }
    _write_json(output_dir / "metrics.json", metrics)
    _write_trace_csv(trace_csv_path, trace_rows)
    print("")
    print(json.dumps(metrics["results"]["all"], indent=2))
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
