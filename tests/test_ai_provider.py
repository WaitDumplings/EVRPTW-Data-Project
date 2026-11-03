"""Provider-boundary tests use synthetic CLI responses, never a real model."""

import json
from pathlib import Path
import subprocess

import pytest

from evrptw_cle.ai_provider import AIProviderConfig, AIProviderError, ClaudeCLIProvider


def annotation(entity_id="building-1"):
    return {"entity_id": entity_id, "ai_inferred_use": "unknown",
            "reason": "Only a generic building tag was supplied.", "evidence": []}


def successful_cli(output):
    events = [
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "w1",
            "name": "WebSearch", "input": {"query": "public building address"}}]}},
        {"type": "result", "is_error": False, "subtype": "success",
         "structured_output": output, "modelUsage": {"test-model": {"inputTokens": 10}}},
    ]
    return subprocess.CompletedProcess([], 0, "\n".join(json.dumps(x) for x in events), "")


def test_real_cli_boundary_is_read_only_and_success_cache_retains_trace(tmp_path, monkeypatch):
    calls = []

    def invoke(command, **kwargs):
        working = Path(kwargs["cwd"])
        assert working.exists() and not list(working.iterdir())
        assert working != tmp_path and not working.is_relative_to(tmp_path)
        calls.append((command, kwargs))
        return successful_cli(annotation())

    monkeypatch.setattr(subprocess, "run", invoke)
    provider = ClaudeCLIProvider(AIProviderConfig(tmp_path))
    payload = {"entity_id": "building-1", "longitude": -96, "osm_objects": []}
    first = provider.run("residential_use", payload)
    second = provider.run("residential_use", payload)
    assert len(calls) == 1 and not first["cache_hit"] and second["cache_hit"]
    command, kwargs = calls[0]
    assert command[command.index("--tools") + 1] == "WebSearch,WebFetch"
    assert command[command.index("--permission-mode") + 1] == "dontAsk"
    assert "--model" not in command
    assert "--strict-mcp-config" in command and "--no-session-persistence" in command
    assert command[command.index("--setting-sources") + 1] == "user"
    assert "--bare" not in command  # Subscription OAuth/keychain remain usable.
    settings = json.loads(command[command.index("--settings") + 1])
    assert not settings["autoMemoryEnabled"]
    assert settings["env"]["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
    assert kwargs["env"]["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert kwargs["env"]["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
    assert kwargs["input"].endswith(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    assert "shell" not in kwargs
    record = json.loads(Path(first["record_path"]).read_text())
    assert record["request"]["payload"] == payload
    assert record["request"]["provider_version"] == "customer_claude_cli_v2_isolated"
    assert not Path(record["isolated_working_directory"]).exists()
    assert record["context_loading"] == {"settings_sources": ["user"],
        "claude_md": False, "auto_memory": False, "project_settings": False}
    assert record["events"][0]["message"]["content"][0]["name"] == "WebSearch"
    assert first["model"] == "test-model"


def test_failed_login_never_becomes_success_and_keeps_actual_failure(tmp_path, monkeypatch):
    failure = {"type": "result", "is_error": True, "result": "Not logged in"}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess([], 1, json.dumps(failure), ""))
    provider = ClaudeCLIProvider(AIProviderConfig(tmp_path))
    with pytest.raises(AIProviderError, match="Not logged in") as exc:
        provider.run("residential_use", {"entity_id": "building-1"})
    record = json.loads(exc.value.record_path.read_text())
    assert record["status"] == "failed" and record["cli_result"] == failure
    assert len(list(tmp_path.glob("residential_use/*.failed.*.json"))) == 1
    assert len(list(tmp_path.glob("residential_use/*.json"))) == 1


def test_unit_imputation_does_not_promote_complex_count_to_building(tmp_path, monkeypatch):
    output = {"entity_id": "building-1", "reason": "Complex reports 10871 units.",
              "units": 10871, "units_low": 10871, "units_high": 10871,
              "units_method": "web_extracted", "units_scope": "complex",
              "evidence": [{"kind": "web", "url": "https://example.org/complex",
                            "scope": "complex", "quote": "10871 units",
                            "source_id": "complex-1", "source_date": None}]}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: successful_cli(output))
    provider = ClaudeCLIProvider(AIProviderConfig(tmp_path))
    with pytest.raises(AIProviderError, match="building scope"):
        provider.run("unit_impute", {"entity_id": "building-1"})


def test_web_facts_and_model_estimates_have_distinct_requirements(tmp_path, monkeypatch):
    output = {"entity_id": "building-1", "reason": "Similar reviewed houses support one unit.",
              "units": 1, "units_low": 1, "units_high": 2,
              "units_method": "model_imputed", "units_scope": "building",
              "evidence": [{"kind": "reference_case", "source_id": "reviewed-1",
                            "url": None, "quote": "One unit", "scope": "building",
                            "source_date": None}]}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: successful_cli(output))
    provider = ClaudeCLIProvider(AIProviderConfig(tmp_path, allow_web=False))
    result = provider.run("unit_impute", {"entity_id": "building-1"})
    assert result["output"]["units_method"] == "model_imputed"
    output["units_method"] = "web_extracted"
    output["entity_id"] = "building-2"
    with pytest.raises(AIProviderError, match="building-level web evidence"):
        provider.run("unit_impute", {"entity_id": "building-2"})


def test_envelope_result_json_and_bounded_serial_batch(tmp_path, monkeypatch):
    calls = []

    def invoke(*args, **kwargs):
        calls.append(kwargs)
        # Compatibility with older ordinary-JSON Claude envelopes.
        raw = {"type": "result", "result": json.dumps(annotation()), "model": "old-model"}
        return subprocess.CompletedProcess([], 0, json.dumps(raw), "")

    monkeypatch.setattr(subprocess, "run", invoke)
    provider = ClaudeCLIProvider(AIProviderConfig(tmp_path))
    requests = [{"task": "residential_use", "payload": {"entity_id": "building-1"}}] * 4
    results = list(provider.run_batch(requests, limit=1))
    assert len(results) == len(calls) == 1 and results[0]["model"] == "old-model"


def test_timeout_records_failure_without_an_annotation(tmp_path, monkeypatch):
    def invoke(*args, **kwargs):
        raise subprocess.TimeoutExpired("claude", 1, output=b"partial event")

    monkeypatch.setattr(subprocess, "run", invoke)
    provider = ClaudeCLIProvider(AIProviderConfig(tmp_path, timeout_seconds=1))
    with pytest.raises(AIProviderError) as exc:
        provider.run("residential_use", {"entity_id": "building-1"})
    record = json.loads(exc.value.record_path.read_text())
    assert record["error_type"] == "TimeoutExpired" and "output" not in record


def test_main_model_from_init_is_separate_from_auxiliary_usage_and_old_cache(tmp_path, monkeypatch):
    calls = []
    events = [{"type": "system", "subtype": "init", "model": "claude-main"},
        {"type": "result", "is_error": False, "structured_output": annotation(),
         "modelUsage": {"claude-main": {}, "claude-auxiliary": {}}}]

    def invoke(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess([], 0, "\n".join(json.dumps(x) for x in events), "")

    monkeypatch.setattr(subprocess, "run", invoke)
    provider = ClaudeCLIProvider(AIProviderConfig(tmp_path))
    first = provider.run("residential_use", {"entity_id": "building-1"})
    assert first["model"] == "claude-main" and first["model_identity_source"] == "cli_init"
    assert first["models"] == ["claude-main", "claude-auxiliary"]
    path = Path(first["record_path"])
    old = json.loads(path.read_text())
    old["model"] = "cli_default_unreported"
    old.pop("model_identity_source", None)
    path.write_text(json.dumps(old))
    before = path.read_bytes()
    cached = provider.run("residential_use", {"entity_id": "building-1"})
    assert len(calls) == 1 and cached["cache_hit"]
    assert cached["model"] == "claude-main" and cached["model_identity_source"] == "cli_init"
    assert path.read_bytes() == before
