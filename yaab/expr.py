"""A small, sandboxed expression language for conditions.

A ``when=``/``stop=`` guard may be written as a string like ``"state.intent ==
'refund'"`` or ``"output.score >= 0.9"``. This module compiles such a string
into the *same* ``Guard -> bool`` function a Python callable would produce, over
the *same* :class:`~yaab.conditions.Guard`, so swapping a string for the
equivalent callable never changes the data it can reach or how it behaves.

Safety is the whole point. The string is parsed with Python's :mod:`ast` in
``eval`` mode and then validated against a **strict node allowlist**: only
comparisons, boolean ``and``/``or``/``not``, membership (``in``), simple
arithmetic, attribute reads on five named roots, and subscript reads are
permitted. There is **no** ``eval``/``exec``, no function calls, no lambda, no
comprehensions, and no attribute traversal to dunders — anything outside the
grammar is rejected at compile time with a clear error. A compiled expression is
therefore *pure data*: it is safe to accept from an untrusted config file.

Reachable roots (identical to what a callable reaches via ``(value, ctx)``):

* ``input``  — the guarded unit's input (only under an INPUT-phase guard);
* ``output`` / ``value`` — the unit's output (only under an OUTPUT-phase guard);
* ``state``  — the run's read-only :class:`~yaab.state.ReadonlyState`;
* ``deps``   — the injected dependency object (attribute reads only);
* ``ctx``    — ``ctx.identity`` / ``ctx.usage`` / ``ctx.run_id`` /
  ``ctx.session_id`` (read-only).

A phase mismatch — referencing ``output`` in an INPUT-phase expression, or
``input`` in an OUTPUT-phase one — is a **compile-time** error, not a silent
mis-evaluation against the wrong value.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .conditions import Condition, Guard

# The five reachable roots (``value`` is an alias of ``output``).
_ROOTS = frozenset({"input", "output", "value", "state", "deps", "ctx"})
_OUTPUT_ROOTS = frozenset({"output", "value"})
_CTX_FIELDS = frozenset({"identity", "usage", "run_id", "session_id"})

# A sentinel for a reference that could not be resolved (a missing state key, a
# missing attribute). It compares unequal to everything real, so a guard over a
# missing value is simply False rather than an exception.
_MISSING: Any = object()

# Bound the work a ``matches`` regex can do (length cap keeps it cheap/safe).
_MAX_MATCH_LEN = 10_000


def _rewrite_word_operators(expr: str) -> str:
    """Rewrite the two word operators into AST-representable binary operators.

    ``contains`` and ``matches`` are not Python operators, so the grammar maps
    them onto two operator symbols Python *can* parse but that are otherwise
    disallowed (``@`` and ``>>``); the compiler then lowers those back to the
    safe substring/regex helpers. The substitution is whole-word only, so a
    string literal like ``'matches'`` or a state key is untouched.
    """
    expr = re.sub(r"(?<![\w.])contains(?![\w.])", " @ ", expr)
    expr = re.sub(r"(?<![\w.])matches(?![\w.])", " >> ", expr)
    # JSON/YAML-friendly literal spellings map to Python's constants.
    expr = re.sub(r"(?<![\w.'\"])true(?![\w.])", "True", expr)
    expr = re.sub(r"(?<![\w.'\"])false(?![\w.])", "False", expr)
    expr = re.sub(r"(?<![\w.'\"])null(?![\w.])", "None", expr)
    return expr


class ConditionSyntaxError(ValueError):
    """A guard expression used a construct outside the safe grammar.

    Raised at compile time for anything not on the allowlist — a function call,
    a lambda, a comprehension, a dunder attribute, an unknown root name, etc.
    The message names the offending construct so a typo or an unsafe expression
    fails loudly rather than being silently accepted or mis-evaluated.
    """


class ConditionPhaseError(ValueError):
    """A guard expression referenced a value its phase cannot bind.

    An INPUT-phase guard (``when=``) runs *before* the unit produces output, so
    it may not reference ``output``/``value``; an OUTPUT-phase guard (``stop=``)
    runs *after*, against the output, so it may not reference ``input``. The
    mismatch is caught at compile time (when the spec loads), not at run time.
    """


def compile_expr(expr: str, *, phase: str) -> Condition:
    """Compile a safe expression string into a :class:`~yaab.conditions.Condition`.

    ``phase`` is supplied by the call site (``when=`` passes the input phase,
    ``stop=`` the output phase); it fixes which value ``input``/``output`` bind
    to and rejects phase-mismatched references. The returned condition carries a
    *probe* that records the resolved operand values for decision events.
    """
    from .conditions import Condition, Phase  # local import avoids a cycle

    source = _rewrite_word_operators(expr)
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ConditionSyntaxError(f"invalid guard expression {expr!r}: {exc.msg}") from exc

    is_output = phase == Phase.OUTPUT
    _validate(tree.body, expr=expr, is_output=is_output)

    fn = _build_eval(tree.body)
    probe = _build_probe(tree.body)

    def _check(guard: Guard) -> bool:
        return bool(fn(guard))

    return Condition(_check, expr=expr, probe=probe)


# --- validation (the allowlist + phase check) ------------------------------


def _validate(node: ast.AST, *, expr: str, is_output: bool) -> None:
    """Reject any node outside the safe grammar (and any phase mismatch)."""
    if isinstance(node, ast.BoolOp):
        for value in node.values:
            _validate(value, expr=expr, is_output=is_output)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.Not, ast.USub, ast.UAdd)):
            raise ConditionSyntaxError(
                f"operator {type(node.op).__name__} is not allowed in a guard expression"
            )
        _validate(node.operand, expr=expr, is_output=is_output)
        return
    if isinstance(node, ast.BinOp):
        # MatMult/RShift are the internal lowerings of ``contains``/``matches``.
        allowed_bin = (
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.FloorDiv,
            ast.MatMult,
            ast.RShift,
        )
        if not isinstance(node.op, allowed_bin):
            raise ConditionSyntaxError(
                f"arithmetic operator {type(node.op).__name__} is not allowed in a guard expression"
            )
        _validate(node.left, expr=expr, is_output=is_output)
        _validate(node.right, expr=expr, is_output=is_output)
        return
    if isinstance(node, ast.Compare):
        for op in node.ops:
            if not isinstance(
                op,
                (
                    ast.Eq,
                    ast.NotEq,
                    ast.Lt,
                    ast.LtE,
                    ast.Gt,
                    ast.GtE,
                    ast.In,
                    ast.NotIn,
                ),
            ):
                raise ConditionSyntaxError(
                    f"comparison {type(op).__name__} is not allowed in a guard expression"
                )
        _validate(node.left, expr=expr, is_output=is_output)
        for comparator in node.comparators:
            _validate(comparator, expr=expr, is_output=is_output)
        return
    if isinstance(node, ast.Constant):
        return
    if isinstance(node, (ast.List, ast.Tuple)):
        for elt in node.elts:
            _validate(elt, expr=expr, is_output=is_output)
        return
    if isinstance(node, ast.Name):
        if node.id not in _ROOTS:
            raise ConditionSyntaxError(
                f"unknown name {node.id!r} in guard expression {expr!r}; "
                f"only {sorted(_ROOTS)} are reachable"
            )
        _check_phase(node.id, is_output=is_output, expr=expr)
        return
    if isinstance(node, ast.Attribute):
        if node.attr.startswith("_"):
            raise ConditionSyntaxError(
                f"attribute {node.attr!r} is not allowed in a guard expression"
            )
        root = _root_name(node)
        if root is None:
            raise ConditionSyntaxError(
                f"attribute access on a non-root expression is not allowed: {expr!r}"
            )
        _check_phase(root, is_output=is_output, expr=expr)
        # ctx exposes only an explicit, read-only field set.
        if root == "ctx" and _is_direct_ctx_field(node) and node.attr not in _CTX_FIELDS:
            raise ConditionSyntaxError(
                f"ctx.{node.attr} is not reachable; allowed: {sorted(_CTX_FIELDS)}"
            )
        _validate(node.value, expr=expr, is_output=is_output)
        return
    if isinstance(node, ast.Subscript):
        root = _root_name(node)
        if root is None:
            raise ConditionSyntaxError(
                f"subscript on a non-root expression is not allowed: {expr!r}"
            )
        _check_phase(root, is_output=is_output, expr=expr)
        key = node.slice
        if not isinstance(key, ast.Constant):
            raise ConditionSyntaxError(
                f"subscript key must be a literal in a guard expression: {expr!r}"
            )
        _validate(node.value, expr=expr, is_output=is_output)
        return
    raise ConditionSyntaxError(
        f"construct {type(node).__name__} is not allowed in a guard expression: {expr!r}"
    )


def _check_phase(root: str, *, is_output: bool, expr: str) -> None:
    if root in _OUTPUT_ROOTS and not is_output:
        raise ConditionPhaseError(
            f"{root!r} is only reachable in an output guard (stop=); "
            f"an input guard (when=) runs before any output exists: {expr!r}"
        )
    if root == "input" and is_output:
        raise ConditionPhaseError(
            f"'input' is only reachable in an input guard (when=); "
            f"an output guard (stop=) sees the output, not the input: {expr!r}"
        )


def _is_direct_ctx_field(node: ast.Attribute) -> bool:
    """True when ``node`` is ``ctx.<field>`` (the value is the ``ctx`` Name)."""
    return isinstance(node.value, ast.Name) and node.value.id == "ctx"


def _root_name(node: ast.AST) -> str | None:
    """The root Name id of an attribute/subscript chain, or None."""
    cur: ast.AST = node
    while isinstance(cur, (ast.Attribute, ast.Subscript)):
        cur = cur.value
    if isinstance(cur, ast.Name):
        return cur.id
    return None


# --- evaluation (build a Guard -> value function) --------------------------


def _build_eval(node: ast.AST) -> Callable[[Guard], Any]:
    """Compile an AST node into a function of the Guard producing its value."""
    if isinstance(node, ast.BoolOp):
        parts = [_build_eval(v) for v in node.values]
        if isinstance(node.op, ast.And):

            def _and(g: Guard) -> Any:
                result: Any = True
                for part in parts:
                    result = part(g)
                    if not result:
                        return result
                return result

            return _and

        def _or(g: Guard) -> Any:
            result: Any = False
            for part in parts:
                result = part(g)
                if result:
                    return result
            return result

        return _or

    if isinstance(node, ast.UnaryOp):
        operand = _build_eval(node.operand)
        if isinstance(node.op, ast.Not):
            return lambda g: not operand(g)
        if isinstance(node.op, ast.USub):
            return lambda g: -operand(g)
        return lambda g: +operand(g)

    if isinstance(node, ast.BinOp):
        left = _build_eval(node.left)
        right = _build_eval(node.right)
        op = node.op
        return lambda g: _binop(op, left(g), right(g))

    if isinstance(node, ast.Compare):
        left = _build_eval(node.left)
        comparators = [_build_eval(c) for c in node.comparators]
        ops = node.ops
        return lambda g: _compare_chain(left(g), ops, comparators, g)

    if isinstance(node, ast.Constant):
        value = node.value
        return lambda g: value

    if isinstance(node, ast.List):
        elts = [_build_eval(e) for e in node.elts]
        return lambda g: [e(g) for e in elts]

    if isinstance(node, ast.Tuple):
        elts = [_build_eval(e) for e in node.elts]
        return lambda g: tuple(e(g) for e in elts)

    if isinstance(node, ast.Name):
        return _build_root(node.id)

    if isinstance(node, ast.Attribute):
        base = _build_eval(node.value)
        attr = node.attr
        return lambda g: _getattr_safe(base(g), attr)

    if isinstance(node, ast.Subscript):
        base = _build_eval(node.value)
        key = node.slice
        assert isinstance(key, ast.Constant)
        key_value = key.value
        return lambda g: _getitem_safe(base(g), key_value)

    raise ConditionSyntaxError(f"cannot evaluate {type(node).__name__}")


def _build_root(name: str) -> Callable[[Guard], Any]:
    if name == "input":
        return lambda g: g.value
    if name in _OUTPUT_ROOTS:
        return lambda g: g.value
    if name == "state":
        return lambda g: g.state
    if name == "deps":
        return lambda g: g.ctx.deps
    if name == "ctx":
        return lambda g: _CtxProxy(g.ctx)
    raise ConditionSyntaxError(f"unknown root {name!r}")


class _CtxProxy:
    """A read-only view exposing only the whitelisted ctx fields."""

    __slots__ = ("_ctx",)

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx

    def __getattr__(self, name: str) -> Any:
        if name in _CTX_FIELDS:
            return getattr(self._ctx, name, _MISSING)
        return _MISSING


def _getattr_safe(obj: Any, attr: str) -> Any:
    if obj is _MISSING:
        return _MISSING
    # Mappings (incl. ReadonlyState) expose values by key, not attribute, so
    # ``state.intent`` reads ``state["intent"]`` — the dotted-path convenience.
    if hasattr(obj, "__getitem__") and not isinstance(obj, (str, bytes)):
        try:
            return obj[attr]
        except (KeyError, TypeError, IndexError):
            pass
    return getattr(obj, attr, _MISSING)


def _getitem_safe(obj: Any, key: Any) -> Any:
    if obj is _MISSING:
        return _MISSING
    try:
        return obj[key]
    except (KeyError, TypeError, IndexError):
        return _MISSING


def _binop(op: ast.operator, left: Any, right: Any) -> Any:
    # ``contains``/``matches`` lower to MatMult/RShift; they yield a bool and
    # tolerate a missing operand (a guard over absent data is simply False).
    if isinstance(op, ast.MatMult):
        return _contains(left, right)
    if isinstance(op, ast.RShift):
        return _matches(left, right)
    if left is _MISSING or right is _MISSING:
        return _MISSING
    if isinstance(op, ast.Add):
        return left + right
    if isinstance(op, ast.Sub):
        return left - right
    if isinstance(op, ast.Mult):
        return left * right
    if isinstance(op, ast.Div):
        return left / right
    if isinstance(op, ast.Mod):
        return left % right
    if isinstance(op, ast.FloorDiv):
        return left // right
    raise ConditionSyntaxError(f"unsupported arithmetic operator {type(op).__name__}")


def _compare_chain(
    left: Any, ops: list[ast.cmpop], comparators: list[Callable[[Guard], Any]], g: Guard
) -> bool:
    cur = left
    for op, comp in zip(ops, comparators, strict=False):
        right = comp(g)
        if not _compare_one(op, cur, right):
            return False
        cur = right
    return True


def _compare_one(op: ast.cmpop, left: Any, right: Any) -> bool:
    # A missing operand never satisfies a comparison (except an explicit
    # equality/inequality, where it is simply unequal to a real value).
    if isinstance(op, ast.Eq):
        return left is not _MISSING and right is not _MISSING and left == right
    if isinstance(op, ast.NotEq):
        if left is _MISSING or right is _MISSING:
            return True
        return left != right
    if left is _MISSING or right is _MISSING:
        return False
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.Gt):
        return left > right
    if isinstance(op, ast.GtE):
        return left >= right
    if isinstance(op, ast.In):
        return left in right
    if isinstance(op, ast.NotIn):
        return left not in right
    raise ConditionSyntaxError(f"unsupported comparison {type(op).__name__}")


# --- operand capture (for decision events) ---------------------------------


def _build_probe(node: ast.AST) -> Callable[[Guard], dict[str, Any]]:
    """Build a function that records every leaf reference's resolved value.

    The probe walks the same references the evaluator does and reports a
    ``{source_text: resolved_value}`` map, so a decision event answers *why* a
    guard fired or skipped, not merely the boolean.
    """
    refs = list(_collect_refs(node))

    def _probe(g: Guard) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for src, fn in refs:
            try:
                value = fn(g)
            except Exception:  # noqa: BLE001 - probing is best-effort, never fatal
                continue
            if value is _MISSING:
                value = None
            out[src] = _jsonish(value)
        return out

    return _probe


def _collect_refs(node: ast.AST) -> Any:
    """Yield ``(source_text, eval_fn)`` for each reference/literal leaf."""
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        if _root_name(node) is not None:
            yield (_unparse(node), _build_eval(node))
            return
    if isinstance(node, ast.Constant):
        yield (_unparse(node), _build_eval(node))
        return
    for child in ast.iter_child_nodes(node):
        yield from _collect_refs(child)


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001
        return type(node).__name__


def _jsonish(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonish(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:  # noqa: BLE001
            return repr(value)
    return repr(value)


def _contains(haystack: Any, needle: Any) -> bool:
    """Lower a ``contains`` operator to a safe substring/membership test."""
    if haystack is _MISSING or needle is _MISSING:
        return False
    try:
        return needle in haystack
    except TypeError:
        return False


def _matches(value: Any, pattern: Any) -> bool:
    """Lower a ``matches`` operator to a bounded, pre-compiled regex search."""
    if value is _MISSING or not isinstance(value, str) or not isinstance(pattern, str):
        return False
    if len(value) > _MAX_MATCH_LEN:
        return False
    try:
        return re.search(pattern, value) is not None
    except re.error:
        return False


__all__ = [
    "compile_expr",
    "ConditionSyntaxError",
    "ConditionPhaseError",
]
