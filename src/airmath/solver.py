"""Tokenize and evaluate a written math expression.

The recognizer produces a flat stream of tokens in left-to-right x order,
e.g. ["1", "2", "+", "8", "="].  Consecutive digit glyphs merge into
multi-digit numbers, and the stream must contain exactly one operator:
"12+8=" -> 20.  Multi-term expressions with precedence are out of scope
for air writing - one gesture, one clean computation.
"""

from __future__ import annotations

OPERATORS = {"+", "-", "*", "/"}


class ExpressionError(ValueError):
    """Raised when the written tokens do not form a computable expression."""


def tokens_to_expression(tokens: list[str]) -> tuple[int, int]:
    """Merge digits into numbers and return (left_operand, right_operand) for
    the *first* operator-joined pair ending at "=" - the common two-term case."""
    numbers, op = merge_tokens(tokens)
    if op is None:
        raise ExpressionError("no operator written")
    if len(numbers) != 2:
        raise ExpressionError("expected two numbers around one operator, got %d" % len(numbers))
    return numbers[0], numbers[1]


def merge_tokens(tokens: list[str]) -> tuple[list[int], str | None]:
    """Return ([numbers...], operator). Digits squashed left-to-right into
    numbers; one operator expected between them. Extra operators raise."""
    numbers: list[int] = []
    pending: list[str] = []
    ops: list[str] = []
    for tok in tokens:
        if tok.isdigit():
            pending.append(tok)
        elif tok in OPERATORS:
            if not pending:
                raise ExpressionError("operator with no left operand")
            numbers.append(int("".join(pending)))
            pending = []
            ops.append(tok)
        else:  # "=" or anything unknown - ignore
            if pending:
                numbers.append(int("".join(pending)))
                pending = []
    if pending:
        numbers.append(int("".join(pending)))
    if len(ops) > 1:
        raise ExpressionError("only one operator per expression for now")
    return numbers, ops[0] if ops else None


def solve(tokens: list[str]) -> int:
    """Evaluate the token stream. Raises ExpressionError on garbage."""
    a, b = tokens_to_expression(tokens)
    op = _first_operator(tokens)
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if b == 0:
        raise ExpressionError("division by zero")
    return a // b


def _first_operator(tokens: list[str]) -> str:
    for tok in tokens:
        if tok in OPERATORS:
            return tok
    raise ExpressionError("no operator written")
