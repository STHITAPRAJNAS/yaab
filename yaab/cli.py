"""The ``yaab`` command-line interface.

Dependency-light (argparse only) so it works on a bare install. Commands:

* ``yaab info``               — environment + active performance backend;
* ``yaab init <name>``        — scaffold a starter agent file;
* ``yaab registry list``      — show the model inventory;
* ``yaab compliance report``  — generate a compliance report for a regime;
* ``yaab serve <module:agent>`` — serve an agent over HTTP (FastAPI).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Optional

from . import __version__, _core


def _info() -> int:
    import platform

    print(f"YAAB {__version__}")
    print(f"  performance backend : {_core.backend()}")
    print(f"  python              : {platform.python_version()}")
    try:
        import yaab_core  # noqa: F401

        print(f"  yaab-core (rust)    : {yaab_core.__version__}")
    except ImportError:
        print("  yaab-core (rust)    : not installed (pure-Python fallback active)")
    return 0


_STARTER = '''"""A starter YAAB agent."""

from yaab import Agent


agent = Agent(
    "{name}",
    model="openai/gpt-4o",
    instructions="You are a helpful assistant.",
)


if __name__ == "__main__":
    print(agent.run_sync("Say hello in one sentence.").output)
'''


def _init(name: str) -> int:
    path = f"{name}.py"
    with open(path, "w") as fh:
        fh.write(_STARTER.format(name=name))
    print(f"created {path}")
    return 0


def _load_attr(spec: str):
    module_name, _, attr = spec.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr or "agent")


def _registry_list(db: Optional[str]) -> int:
    from .governance.registry import AgentRegistry, SQLiteRegistryBackend

    registry = AgentRegistry(SQLiteRegistryBackend(db) if db else None)
    rows = registry.inventory()
    if not rows:
        print("(registry is empty)")
        return 0
    for row in rows:
        print(
            f"{row['agent_id']:<24} {row['name']:<20} "
            f"tier={row['risk_tier']:<8} status={row['approval_status']:<10} "
            f"state={row['lifecycle_state']}"
        )
    return 0


def _compliance_report(regime: str, db: Optional[str], agent_id: Optional[str]) -> int:
    from .governance.audit import AuditLog
    from .governance.compliance import get_mapper
    from .governance.registry import AgentRegistry, SQLiteRegistryBackend

    mapper = get_mapper(regime)
    if mapper is None:
        print(f"unknown regime '{regime}'. Try: sr_11_7, eu_ai_act, nist_ai_rmf, iso_42001, soc2")
        return 1
    registry = AgentRegistry(SQLiteRegistryBackend(db) if db else None)
    report = mapper.map(registry, AuditLog(), agent_id)
    print(report.to_markdown())
    return 0


def _serve(spec: str, host: str, port: int) -> int:
    from .serve import serve

    serve(_load_attr(spec), host=host, port=port)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="yaab", description="Yet Another Agent Builder")
    parser.add_argument("--version", action="version", version=f"yaab {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="show environment and performance backend")

    p_init = sub.add_parser("init", help="scaffold a starter agent")
    p_init.add_argument("name")

    p_reg = sub.add_parser("registry", help="agent registry commands")
    reg_sub = p_reg.add_subparsers(dest="registry_command")
    p_reg_list = reg_sub.add_parser("list", help="show the model inventory")
    p_reg_list.add_argument("--db", default=None)

    p_comp = sub.add_parser("compliance", help="compliance commands")
    comp_sub = p_comp.add_subparsers(dest="compliance_command")
    p_comp_report = comp_sub.add_parser("report", help="generate a compliance report")
    p_comp_report.add_argument("regime")
    p_comp_report.add_argument("--db", default=None)
    p_comp_report.add_argument("--agent-id", default=None)

    p_serve = sub.add_parser("serve", help="serve an agent over HTTP (module:agent)")
    p_serve.add_argument("spec")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.command == "info" or args.command is None:
        return _info()
    if args.command == "init":
        return _init(args.name)
    if args.command == "registry":
        if args.registry_command == "list":
            return _registry_list(args.db)
        print("usage: yaab registry list [--db PATH]")
        return 1
    if args.command == "compliance":
        if args.compliance_command == "report":
            return _compliance_report(args.regime, args.db, args.agent_id)
        print("usage: yaab compliance report <regime> [--db PATH] [--agent-id ID]")
        return 1
    if args.command == "serve":
        return _serve(args.spec, args.host, args.port)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
