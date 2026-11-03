"""Independent GPT building judge through an authenticated local Codex CLI.

Each request has an empty temporary working directory and receives only the
shared payload/prompt. Saved Claude/GPT judgments are outside this directory.
This provider records proposals; domain acceptance remains in enrichment code.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import tomllib
from typing import Any, Iterable, Iterator

from .ai_provider import (AIProviderError, _dumps, _structured_output, _timeout_text,
                          _validate_output, _write, default_prompt, output_schema)

PROVIDER_VERSION = "customer_codex_cli_v1"
BUNDLED_CODEX = "/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex"


@dataclass(frozen=True)
class GPTProviderConfig:
    cache_dir: Path
    command: str = "codex"
    model: str | None = None
    timeout_seconds: float = 600
    allow_web: bool = True
    prompt_version: str = "customer_evidence_v1"


def _executable(command):
    found = shutil.which(str(command))
    if found:
        return found
    if str(command) == "codex" and Path(BUNDLED_CODEX).is_file():
        return BUNDLED_CODEX
    return str(command)


def _configured_model():
    """Read only the model identifier; never copy user config or credentials."""
    directory = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    try:
        value = tomllib.loads((directory / "config.toml").read_text()).get("model")
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return value.strip() if isinstance(value, str) and value.strip() else None


def _events(stdout):
    if not stdout.strip():
        return []
    events = []
    for line in stdout.splitlines():
        if line.strip():
            event = json.loads(line)
            if isinstance(event, dict):
                events.append(event)
    return events


def _final_annotation(events, answer_path):
    if answer_path.is_file() and answer_path.read_text().strip():
        return _structured_output({"result": answer_path.read_text()})
    for event in reversed(events):
        item = event.get("item", {})
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            return _structured_output({"result": item.get("text", "")})
    raise ValueError("Codex CLI returned no final structured annotation")


def _actual_models(events):
    models = []
    for event in events:
        for scope in [event, event.get("session", {}), event.get("thread", {})]:
            value = scope.get("model") if isinstance(scope, dict) else None
            if isinstance(value, str) and value and value not in models:
                models.append(value)
    return models


class CodexCLIProvider:
    """Same annotation interface as ClaudeCLIProvider, with isolated GPT calls."""

    def __init__(self, config: GPTProviderConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir).expanduser().resolve()
        self.command = _executable(config.command)
        self.model = config.model or _configured_model()
        self._version = None

    def readiness(self):
        if not shutil.which(self.command):
            return dict(available=False, authenticated=False, reason="codex_not_found")
        try:
            version = subprocess.run([self.command, "--version"], capture_output=True,
                                     text=True, timeout=15, check=False)
            auth = subprocess.run([self.command, "login", "status"], capture_output=True,
                                  text=True, timeout=15, check=False)
            self._version = version.stdout.strip()
            status = (auth.stdout + "\n" + auth.stderr).lower()
            authenticated = auth.returncode == 0 and "logged in" in status
            method = "chatgpt" if "chatgpt" in status else "api_key" if "api key" in status else "unknown"
            return dict(available=True, authenticated=authenticated, auth_method=method,
                        version=self._version, model=self.model)
        except (OSError, subprocess.TimeoutExpired) as error:
            return dict(available=True, authenticated=False, reason=type(error).__name__)

    def _command(self, schema_path, answer_path):
        command = [self.command, "exec", "--ignore-user-config", "--ignore-rules",
                   "--skip-git-repo-check", "--ephemeral", "--sandbox", "read-only",
                   "--json", "--color", "never", "--output-schema", str(schema_path),
                   "--output-last-message", str(answer_path)]
        # No user/project MCP configuration, shell, connectors, memory, skills,
        # browser/computer interaction or other judges' files are model inputs.
        overrides = {
            "web_search": "live" if self.config.allow_web else "disabled",
            "approval_policy": "never", "mcp_servers": {}, "project_doc_max_bytes": 0,
            "features.shell_tool": False, "features.apps": False,
            "features.plugins": False, "features.memories": False,
            "features.multi_agent": False, "features.hooks": False,
            "features.browser_use": False, "features.computer_use": False,
            "features.image_generation": False, "features.view_image": False,
            "features.code_mode": False, "features.goals": False,
            "skills.bundled.enabled": False, "skills.include_instructions": False,
        }
        for key, value in overrides.items():
            command.extend(["-c", key + "=" + _dumps(value)])
        if self.model:
            command.extend(["--model", self.model])
        command.append("-")
        return command

    def run(self, task, payload, *, prompt=None):
        schema = output_schema(task)
        if not payload.get("entity_id"):
            raise ValueError("AI request requires entity_id")
        prompt = default_prompt(task) if prompt is None else prompt
        request = dict(provider_version=PROVIDER_VERSION, task=task, payload=payload,
                       prompt=prompt, schema=schema, prompt_version=self.config.prompt_version,
                       requested_model=self.model, allow_web=self.config.allow_web)
        request_id = hashlib.sha256(_dumps(request).encode()).hexdigest()
        record_path = self.cache_dir / task / (request_id + ".json")
        if record_path.is_file():
            record = json.loads(record_path.read_text())
            if record.get("status") == "success":
                return self._result(record, record_path, cache_hit=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        record = dict(request_id=request_id, request=request,
                      started_at_utc=datetime.now(UTC).isoformat(), provider="codex_cli",
                      cli_version=self._version, status="failed")
        text = prompt + "\n\nINPUT_JSON:\n" + _dumps(payload)
        try:
            if self._version is None:
                version = subprocess.run([self.command, "--version"], capture_output=True,
                                         text=True, timeout=15, check=False)
                self._version = version.stdout.strip()
                record["cli_version"] = self._version
            with TemporaryDirectory(prefix="evrptw-gpt-judge-") as directory:
                work = Path(directory)
                schema_path, answer_path = work / "schema.json", work / "answer.json"
                _write(schema_path, schema)
                command = self._command(schema_path, answer_path)
                record.update(command=command, isolated_working_directory=str(work),
                              input_sha256=hashlib.sha256(text.encode()).hexdigest())
                process = subprocess.run(command, input=text, capture_output=True, text=True,
                    cwd=work, timeout=self.config.timeout_seconds, check=False)
                record.update(exit_code=process.returncode, raw_stdout=process.stdout,
                              raw_stderr=process.stderr)
                events = _events(process.stdout)
                record["events"] = events
                failures = [event for event in events if event.get("type") == "turn.failed"]
                if process.returncode or failures:
                    message = failures[-1].get("error", {}) if failures else process.stderr.strip()
                    raise ValueError(str(message or f"Codex exited {process.returncode}"))
                forbidden = [event for event in events if event.get("item", {}).get("type") in
                    {"command_execution", "mcp_tool_call", "file_change", "computer_use", "browser_use", "image_generation"}]
                if forbidden:
                    raise ValueError("GPT judge attempted a tool outside the public-web annotation scope")
                output = _final_annotation(events, answer_path)
                _validate_output(task, output, payload)
            finished = datetime.now(UTC).isoformat()
            for evidence in output["evidence"]:
                if evidence.get("url"):
                    evidence["recorded_at_utc"] = finished
            models = _actual_models(events)
            model = models[0] if len(models) == 1 else self.model or "cli_default_unreported"
            record.update(status="success", completed_at_utc=finished, output=output,
                          models=models or ([self.model] if self.model else []), model=model,
                          model_identity_source="cli_event" if models else "explicit_cli_model" if self.model else "unreported")
        except (OSError, subprocess.TimeoutExpired, ValueError, TypeError) as error:
            if isinstance(error, subprocess.TimeoutExpired):
                record["raw_stdout"] = _timeout_text(error.stdout)
                record["raw_stderr"] = _timeout_text(error.stderr)
            record.update(error_type=type(error).__name__, error=str(error),
                          completed_at_utc=datetime.now(UTC).isoformat())
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            failure_path = record_path.with_name(record_path.stem + f".failed.{stamp}.json")
            _write(failure_path, record)
            raise AIProviderError(f"GPT {task} request failed: {error}", record_path=failure_path) from error
        _write(record_path, record)
        return self._result(record, record_path, cache_hit=False)

    @staticmethod
    def _result(record, record_path, *, cache_hit):
        return dict(request_id=record["request_id"], task=record["request"]["task"],
                    output=record["output"], model=record["model"], models=record["models"],
                    evidence=record["output"]["evidence"], cache_hit=cache_hit,
                    record_path=str(record_path))

    def run_batch(self, requests: Iterable[dict[str, Any]], *, limit=None) -> Iterator[dict[str, Any]]:
        for index, request in enumerate(requests):
            if limit is not None and index >= limit:
                break
            yield self.run(request["task"], request["payload"], prompt=request.get("prompt"))


GPTCLIProvider = CodexCLIProvider
