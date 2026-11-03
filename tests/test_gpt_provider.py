"""Actual subprocess boundary tests with a local fake CLI; no model requests."""
import json
from pathlib import Path
import sys

import pytest

from evrptw_cle.ai_provider import AIProviderError, default_prompt
from evrptw_cle.gpt_provider import CodexCLIProvider, GPTProviderConfig


def fake_cli(tmp_path, *, output=None, exit_code=0, answer=True, forbidden=False, sleep=0):
    executable = tmp_path / "fake-codex"
    log = tmp_path / "invocations.jsonl"
    if output is None:
        output = dict(entity_id="building-1", ai_inferred_use="unknown",
                      reason="Generic building tag does not establish use.", evidence=[])
    source = f'''#!{sys.executable}
import json,os,sys,time
from pathlib import Path
args=sys.argv[1:]
if args==['--version']:
    print('codex-cli test-version');sys.exit(0)
if args==['login','status']:
    print('Logged in using ChatGPT',file=sys.stderr);sys.exit(0)
text=sys.stdin.read()
with Path({str(log)!r}).open('a') as stream:
    stream.write(json.dumps(dict(argv=args,cwd=os.getcwd(),files=sorted(os.listdir()),stdin=text))+'\\n')
print(json.dumps(dict(type='thread.started',thread_id='independent-test',model='test-gpt')),flush=True)
if {sleep!r}:
    time.sleep({sleep!r})
output=json.loads({json.dumps(output)!r})
if {answer!r}:
    Path(args[args.index('--output-last-message')+1]).write_text(json.dumps(output))
if {forbidden!r}:
    print(json.dumps(dict(type='item.completed',item=dict(type='command_execution',command='cat other-judge.json'))))
print(json.dumps(dict(type='item.completed',item=dict(type='web_search',query='public building evidence'))))
print(json.dumps(dict(type='item.completed',item=dict(type='agent_message',text=json.dumps(output)))))
print(json.dumps(dict(type='turn.completed',usage=dict(input_tokens=20,output_tokens=10))))
sys.exit({exit_code!r})
'''
    executable.write_text(source)
    executable.chmod(0o755)
    return executable, log


def provider(tmp_path, executable, **kwargs):
    return CodexCLIProvider(GPTProviderConfig(tmp_path / "cache", command=str(executable),
                                             model="requested-gpt", **kwargs))


def test_readiness_reports_login_without_account_details(tmp_path):
    executable, _ = fake_cli(tmp_path)
    status = provider(tmp_path, executable).readiness()
    assert status == dict(available=True, authenticated=True, auth_method="chatgpt",
                         version="codex-cli test-version", model="requested-gpt")


def test_independent_isolated_subprocess_keeps_exact_prompt_payload_and_cache(tmp_path):
    executable, log = fake_cli(tmp_path)
    client = provider(tmp_path, executable)
    client.cache_dir.mkdir()
    (client.cache_dir / "other_judges.json").write_text('{"claude":"secret-other-answer"}')
    payload = dict(entity_id="building-1", longitude=-96, latitude=32, osm_objects=[])
    first = client.run("residential_use", payload)
    second = client.run("residential_use", payload)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 1 and not first["cache_hit"] and second["cache_hit"]
    call = calls[0]
    assert call["files"] == ["schema.json"]
    assert not Path(call["cwd"]).is_relative_to(client.cache_dir)
    assert not Path(call["cwd"]).exists()
    assert call["stdin"] == default_prompt("residential_use") + "\n\nINPUT_JSON:\n" + json.dumps(payload,sort_keys=True,separators=(",", ":"))
    assert "secret-other-answer" not in call["stdin"]
    args = call["argv"]
    assert args[args.index("--sandbox")+1] == "read-only"
    assert "--ignore-user-config" in args and "--ephemeral" in args
    assert "features.shell_tool=false" in args and "mcp_servers={}" in args
    assert 'web_search="live"' in args
    assert args[args.index("--model")+1] == "requested-gpt"
    record = json.loads(Path(first["record_path"]).read_text())
    assert record["request"]["payload"] == payload
    assert record["events"][1]["item"]["type"] == "web_search"
    assert record["cli_version"] == "codex-cli test-version"
    assert first["model"] == "test-gpt"


def test_no_web_mode_and_event_final_output_compatibility(tmp_path):
    executable, log = fake_cli(tmp_path, answer=False)
    result = provider(tmp_path, executable, allow_web=False).run("residential_use", {"entity_id":"building-1"})
    assert result["output"]["ai_inferred_use"] == "unknown"
    args = json.loads(log.read_text().splitlines()[0])["argv"]
    assert 'web_search="disabled"' in args


def test_failed_process_keeps_failure_record_not_annotation(tmp_path):
    executable, _ = fake_cli(tmp_path, exit_code=1)
    with pytest.raises(AIProviderError) as failed:
        provider(tmp_path, executable).run("residential_use", {"entity_id":"building-1"})
    record = json.loads(failed.value.record_path.read_text())
    assert record["status"] == "failed" and record["exit_code"] == 1
    assert "output" not in record
    assert list((tmp_path / "cache/residential_use").glob("*.failed.*.json"))


def test_timeout_preserves_actual_partial_cli_output(tmp_path):
    executable, _ = fake_cli(tmp_path, sleep=2)
    with pytest.raises(AIProviderError) as failed:
        provider(tmp_path, executable, timeout_seconds=.3).run("residential_use", {"entity_id":"building-1"})
    record = json.loads(failed.value.record_path.read_text())
    assert record["error_type"] == "TimeoutExpired"
    assert "thread.started" in record["raw_stdout"] and "output" not in record


def test_shell_or_other_local_tools_are_not_accepted_as_judgment(tmp_path):
    executable, _ = fake_cli(tmp_path, forbidden=True)
    with pytest.raises(AIProviderError, match="outside the public-web"):
        provider(tmp_path, executable).run("residential_use", {"entity_id":"building-1"})


def test_building_imputation_cannot_promote_complex_total(tmp_path):
    output = dict(entity_id="building-1", units=10871, units_low=10871, units_high=10871,
        units_method="web_extracted", units_scope="complex", reason="Complex-wide count.",
        evidence=[dict(kind="web",source_id="complex",url="https://example.org/complex",
                       quote="10871 units in complex",scope="complex",source_date=None)])
    executable, _ = fake_cli(tmp_path, output=output)
    with pytest.raises(AIProviderError, match="building scope"):
        provider(tmp_path, executable).run("unit_impute", {"entity_id":"building-1"})


def test_only_user_model_identifier_is_read_and_pinned(tmp_path, monkeypatch):
    executable, log = fake_cli(tmp_path)
    config_dir = tmp_path / "codex-settings"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text('model="configured-gpt"\nsecret="fake-secret-not-to-copy"\n')
    monkeypatch.setenv("CODEX_HOME", str(config_dir))
    client = CodexCLIProvider(GPTProviderConfig(tmp_path / "cache", command=str(executable)))
    result = client.run("residential_use", {"entity_id":"building-1"})
    record = json.loads(Path(result["record_path"]).read_text())
    assert record["request"]["requested_model"] == "configured-gpt"
    assert "fake-secret-not-to-copy" not in Path(result["record_path"]).read_text()
    assert json.loads(log.read_text().splitlines()[0])["argv"][-2:] == ["configured-gpt", "-"]
