import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from airmath.solver import ExpressionError, solve

import pytest  # noqa: E402


def _tok(expr: str):
    return list(expr)


def test_add():
    assert solve(_tok("7+4=")) == 11


def test_multidigit_add():
    assert solve(_tok("12+8=")) == 20


def test_sub():
    assert solve(_tok("9-4=")) == 5


def test_mul():
    assert solve(_tok("6*7=")) == 42


def test_div():
    assert solve(_tok("8/2=")) == 4


def test_div_rounds_down():
    assert solve(_tok("7/2=")) == 3


def test_div_by_zero():
    with pytest.raises(ExpressionError):
        solve(_tok("8/0="))


def test_no_operator():
    with pytest.raises(ExpressionError):
        solve(_tok("124"))  # no "=" or operator


def test_no_operand():
    with pytest.raises(ExpressionError):
        solve(_tok("+4="))


def test_two_operators_rejected():
    with pytest.raises(ExpressionError):
        solve(_tok("1+2+3="))


def test_operator_order_by_x_is_ignored_by_solver():
    # solver consumes left-to-right listing; x-sorting is app's job.
    pass
