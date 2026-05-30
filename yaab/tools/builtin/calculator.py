"""A safe arithmetic calculator tool (no eval, AST-validated)."""

from __future__ import annotations

import ast
import operator

from ..base import tool

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression (e.g. '2 * (3 + 4) ** 2').

    Supports + - * / // % ** and parentheses on numbers only. No variables,
    functions, or names — it is parsed with the AST, never ``eval``.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree.body)
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as exc:
        return f"error: {exc}"
    return str(result)
