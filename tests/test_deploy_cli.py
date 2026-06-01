"""Tests for ``yaab deploy`` — generated + executable deployment artifacts.

The deploy command's whole value is *predictable artifacts*: a Dockerfile that
actually serves the agent, a ``gcloud run deploy`` argv with the right shape, a
``fly.toml`` keyed to the app, and a compose file. So the tests assert on the
load-bearing lines of each generator rather than byte-for-byte content, and they
prove the two safety invariants the spec demands: ``dry_run=True`` (the default)
touches no disk and runs nothing, and secrets only ever appear as placeholders —
never harvested from the local environment into a file we write out.

The ``--execute`` path is exercised with a monkeypatched ``subprocess.run`` so
the test records the argv the CLI *would* have run without ever shelling out to
docker / gcloud / flyctl (which need not be installed in CI).
"""

from __future__ import annotations

import json

import pytest

from yaab import deploy as deploy_mod
from yaab.cli import main
from yaab.deploy import (
    DeployPlan,
    deploy,
    generate_cloud_run_cmd,
    generate_compose,
    generate_dockerfile,
    generate_fly_toml,
)

SPEC = "myapp.main:agent"


# --- generators ------------------------------------------------------------


def test_dockerfile_has_load_bearing_lines():
    df = generate_dockerfile(SPEC)
    assert "FROM python:3.12-slim" in df
    # Installs YAAB with the requested extras.
    assert "pip install" in df
    assert "yaab[litellm,serve]" in df
    # The CMD must serve *this* agent, bound to all interfaces.
    assert "yaab serve" in df
    assert SPEC in df
    assert "--host 0.0.0.0" in df
    assert "EXPOSE 8000" in df


def test_dockerfile_respects_python_version_extras_and_port():
    df = generate_dockerfile(SPEC, python_version="3.11", extras="serve,otel", port=9000)
    assert "FROM python:3.11-slim" in df
    assert "yaab[serve,otel]" in df
    assert "EXPOSE 9000" in df
    assert "--port 9000" in df


def test_cloud_run_cmd_argv_structure():
    argv = generate_cloud_run_cmd(
        "gcr.io/proj/img:latest",
        service_name="my-svc",
        region="europe-west1",
    )
    assert argv[:3] == ["gcloud", "run", "deploy"]
    assert "my-svc" in argv
    # Image + region are passed as flags.
    assert "--image" in argv
    assert "gcr.io/proj/img:latest" in argv
    assert "--region" in argv
    assert "europe-west1" in argv
    # Default is locked down (no public access).
    assert "--no-allow-unauthenticated" in argv
    assert "--allow-unauthenticated" not in argv


def test_cloud_run_cmd_allow_unauthenticated_and_env_placeholders():
    argv = generate_cloud_run_cmd(
        "img",
        service_name="svc",
        env={"GEMINI_API_KEY": "", "LOG_LEVEL": "info"},
        allow_unauthenticated=True,
    )
    assert "--allow-unauthenticated" in argv
    assert "--no-allow-unauthenticated" not in argv
    joined = " ".join(argv)
    # A secret-looking key with no value renders as a placeholder, not a real value.
    assert "GEMINI_API_KEY=<your-key>" in joined
    # A non-secret with an explicit value is passed through verbatim.
    assert "LOG_LEVEL=info" in joined


def test_fly_toml_carries_app_and_port():
    toml = generate_fly_toml("my-fly-app", 8000)
    assert 'app = "my-fly-app"' in toml
    assert "internal_port = 8000" in toml


def test_compose_has_agent_service_and_optional_backends():
    plain = generate_compose(SPEC, 8000)
    assert "services:" in plain
    assert "8000:8000" in plain
    # No backends requested -> none present.
    assert "postgres" not in plain
    assert "redis" not in plain

    with_backends = generate_compose(SPEC, 8000, postgres=True, redis=True)
    assert "postgres" in with_backends
    assert "redis" in with_backends


# --- deploy() plan ---------------------------------------------------------


def test_deploy_docker_dry_run_returns_plan_without_touching_disk(tmp_path, monkeypatch):
    # Run from an empty cwd; assert nothing is written there.
    monkeypatch.chdir(tmp_path)
    plan = deploy("docker", SPEC, dry_run=True)
    assert isinstance(plan, DeployPlan)
    assert plan.target == "docker"
    assert "Dockerfile" in plan.files
    assert plan.commands  # at least a docker build command
    assert plan.commands[0][0] == "docker"
    # dry_run must not write anything.
    assert list(tmp_path.iterdir()) == []


def test_deploy_cloud_run_plan_includes_dockerfile_and_gcloud():
    plan = deploy("cloud-run", SPEC, dry_run=True, service_name="svc")
    assert plan.target == "cloud-run"
    assert "Dockerfile" in plan.files
    # A gcloud command is part of the plan.
    assert any(cmd[:3] == ["gcloud", "run", "deploy"] for cmd in plan.commands)


def test_deploy_fly_plan_emits_fly_toml():
    plan = deploy("fly", SPEC, dry_run=True, app_name="zonk")
    assert plan.target == "fly"
    assert "fly.toml" in plan.files
    assert 'app = "zonk"' in plan.files["fly.toml"]


def test_deploy_unknown_target_raises():
    with pytest.raises(ValueError):
        deploy("kubernetes", SPEC, dry_run=True)


def test_deploy_never_reads_real_secret_from_env(monkeypatch):
    # An obviously-fake sentinel: the test only checks it never leaks into artifacts.
    sentinel = "fake-test-value-" + "do-not-leak"
    monkeypatch.setenv("GEMINI_API_KEY", sentinel)
    plan = deploy("cloud-run", SPEC, dry_run=True, service_name="svc", env={"GEMINI_API_KEY": ""})
    blob = json.dumps(plan.model_dump())
    assert sentinel not in blob
    assert "GEMINI_API_KEY=<your-key>" in blob


def test_deploy_execute_writes_files_and_runs_commands(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []

    class _Completed:
        def __init__(self, args):
            self.args = args
            self.returncode = 0
            self.stdout = "ok"
            self.stderr = ""

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return _Completed(args)

    monkeypatch.setattr(deploy_mod.subprocess, "run", fake_run)

    plan = deploy("docker", SPEC, dry_run=False)
    # Files were actually written to disk this time.
    assert (tmp_path / "Dockerfile").exists()
    # The docker build command was executed (recorded by our fake).
    assert calls
    assert calls[0][0] == "docker"
    # Captured output landed in the plan notes/results.
    assert any("ok" in n for n in plan.notes)


# --- CLI -------------------------------------------------------------------


def test_cli_deploy_plan_only_prints_files_and_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = main(["deploy", "docker", SPEC])
    assert code == 0
    out = capsys.readouterr().out
    # The Dockerfile content is printed.
    assert "FROM python:3.12-slim" in out
    assert "yaab serve" in out
    # The command that *would* run is shown.
    assert "docker" in out
    # Plan-only mode writes nothing.
    assert list(tmp_path.iterdir()) == []


def test_cli_deploy_out_writes_files(tmp_path):
    out_dir = tmp_path / "build"
    code = main(["deploy", "docker", SPEC, "--out", str(out_dir)])
    assert code == 0
    assert (out_dir / "Dockerfile").read_text(encoding="utf-8")


def test_cli_deploy_execute_runs_subprocess(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []

    class _Completed:
        def __init__(self, args):
            self.args = args
            self.returncode = 0
            self.stdout = "built"
            self.stderr = ""

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return _Completed(args)

    monkeypatch.setattr(deploy_mod.subprocess, "run", fake_run)

    code = main(["deploy", "docker", SPEC, "--execute"])
    assert code == 0
    # The deploy actually invoked our fake subprocess (never a real docker).
    assert calls
    assert calls[0][0] == "docker"


def test_cli_deploy_env_and_service_name_flags(tmp_path, capsys):
    code = main(
        [
            "deploy",
            "cloud-run",
            SPEC,
            "--service-name",
            "my-svc",
            "--region",
            "asia-south1",
            "--env",
            "GEMINI_API_KEY=",
            "--env",
            "LOG_LEVEL=debug",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "gcloud" in out
    assert "my-svc" in out
    assert "asia-south1" in out
    # Secret rendered as a placeholder; explicit value passed through.
    assert "GEMINI_API_KEY=<your-key>" in out
    assert "LOG_LEVEL=debug" in out


def test_cli_deploy_secrets_never_show_real_env_values(monkeypatch, capsys):
    # An obviously-fake sentinel: the test only checks it never leaks into output.
    sentinel = "fake-test-value-" + "do-not-leak"
    monkeypatch.setenv("GEMINI_API_KEY", sentinel)
    code = main(["deploy", "cloud-run", SPEC, "--service-name", "svc", "--env", "GEMINI_API_KEY="])
    assert code == 0
    out = capsys.readouterr().out
    assert sentinel not in out
    assert "GEMINI_API_KEY=<your-key>" in out
