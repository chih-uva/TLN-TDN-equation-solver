"""
Utilities for evaluating the L=2 source term of h00.

This module provides:
1) A "full" (non-polynomial) source term parsed from Mathematica text.
2) Analytic series expansions up to r^3, r^5, and r^7 (from text files).

The expressions are stored under:
  high_bg_order_sol/source expression/
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from sympy import lambdify
from sympy.parsing.mathematica import parse_mathematica


_BASE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _BASE_DIR / "source expression"

_FULL_PATH = _SRC_DIR / "source_full.txt"
_R3_PATH = _SRC_DIR / "source_r3.txt"
_R5_PATH = _SRC_DIR / "source_r5.txt"
_R7_PATH = _SRC_DIR / "source_r7.txt"
_R9_PATH = _SRC_DIR / "source_r9.txt"


def _read_expr(path: Path) -> str:
    return path.read_text()


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _sanitize_full_expr(expr_str: str) -> str:
    # Normalize Mathematica tokens to simple names.
    expr_str = expr_str.replace(r"\[Prime]", "Prime")

    simple_tokens = {
        r"\[Lambda]": "lamb",
        r"\[Nu]": "nu",
        r"\[Pi]": "Pi",
        r"\[Rho]0": "rho0",
        r"\[Eta]": "eta",
    }
    for old, new in simple_tokens.items():
        expr_str = expr_str.replace(old, new)

    replacements = [
        (r"Derivative\s*\[\s*1\s*\]\s*\[\s*h00\s*\]\s*\[\s*r\s*\]", "dh00"),
        (r"h00\s*\[\s*r\s*\]", "h00"),
        (r"p0\s*\[\s*r\s*\]", "p0"),
        (r"rho0\s*\[\s*r\s*\]", "rho0"),
        (r"Derivative\s*\[\s*1\s*\]\s*\[\s*rho0\s*\]\s*\[\s*r\s*\]", "drho0"),
        (r"\(\s*rho0\^\s*Prime\s*Prime\s*\)\s*\[\s*r\s*\]", "d2rho0"),
        (r"\(\s*eta\^\s*Prime\s*Prime\s*\)\s*\[\s*rho0\s*\[\s*r\s*\]\s*\]", "d2eta"),
        (r"Derivative\s*\[\s*1\s*\]\s*\[\s*eta\s*\]\s*\[\s*rho0\s*\[\s*r\s*\]\s*\]", "deta"),
        (r"eta\s*\[\s*rho0\s*\[\s*r\s*\]\s*\]", "eta"),
        (r"lamb\s*\[\s*r\s*\]", "lamb"),
        (r"nu\s*\[\s*r\s*\]", "nu"),
    ]
    for pattern, repl in replacements:
        expr_str = re.sub(pattern, repl, expr_str)

    # Clean any remaining eta derivative markers.
    expr_str = expr_str.replace("(eta^[Prime][Prime])[rho0]", "d2eta")
    expr_str = expr_str.replace("eta^PrimePrime", "d2eta")
    expr_str = expr_str.replace("Derivative[ 1][eta][rho0]", "deta")
    expr_str = re.sub(r"d2eta\s*\(\s*rho0\s*\)", "d2eta", expr_str)
    expr_str = re.sub(r"deta\s*\(\s*rho0\s*\)", "deta", expr_str)
    expr_str = re.sub(r"eta\s*\(\s*rho0\s*\)", "eta", expr_str)

    return expr_str


def _sanitize_series_expr(expr_str: str) -> str:
    # Normalize Mathematica tokens for series expressions.
    simple_tokens = {
        r"\[Nu]c": "nu_c",
        r"\[Pi]": "Pi",
    }
    for old, new in simple_tokens.items():
        expr_str = expr_str.replace(old, new)
    return expr_str


def _lambdify_expr(expr_str: str, arg_names: tuple[str, ...]) -> Callable:
    expr = parse_mathematica(expr_str)
    symbols = {name: parse_mathematica(name) for name in arg_names}
    return lambdify(tuple(symbols[name] for name in arg_names), expr, modules=["numpy"])


@lru_cache(maxsize=1)
def _get_full_source_fn() -> Callable:
    raw = _read_expr(_FULL_PATH)
    expr_str = _normalize_whitespace(raw)
    expr_str = _sanitize_full_expr(expr_str)
    args = (
        "r",
        "p0",
        "rho0",
        "drho0",
        "d2rho0",
        "h00",
        "dh00",
        "lamb",
        "nu",
        "eta",
        "deta",
        "d2eta",
    )
    return _lambdify_expr(expr_str, args)


@lru_cache(maxsize=1)
def _get_r3_fn() -> Callable:
    raw = _read_expr(_R3_PATH)
    expr_str = _normalize_whitespace(raw)
    expr_str = _sanitize_series_expr(expr_str)
    args = ("r", "K", "pc", "a0", "nu_c", "T")
    return _lambdify_expr(expr_str, args)


@lru_cache(maxsize=1)
def _get_r5_fn() -> Callable:
    raw = _read_expr(_R5_PATH)
    expr_str = _normalize_whitespace(raw)
    expr_str = _sanitize_series_expr(expr_str)
    args = ("r", "K", "pc", "a0", "nu_c", "T")
    return _lambdify_expr(expr_str, args)


@lru_cache(maxsize=1)
def _get_r7_fn() -> Callable:
    raw = _read_expr(_R7_PATH)
    expr_str = _normalize_whitespace(raw)
    expr_str = _sanitize_series_expr(expr_str)
    args = ("r", "K", "pc", "a0", "nu_c", "T")
    return _lambdify_expr(expr_str, args)


@lru_cache(maxsize=1)
def _get_r9_fn() -> Callable:
    raw = _read_expr(_R9_PATH)
    expr_str = _normalize_whitespace(raw)
    expr_str = _sanitize_series_expr(expr_str)
    args = ("r", "K", "pc", "a0", "nu_c", "T")
    return _lambdify_expr(expr_str, args)


class _CallableArray:
    """
    Wrap an array/scalar so it behaves both as a callable f(rho) and a numeric value.
    This mirrors the behavior used in low_bg_order_sol/source-L2_old.py.
    """

    def __init__(self, value):
        self.value = np.asarray(value)

    def __call__(self, rho):
        return np.broadcast_to(self.value, np.shape(rho))

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)

    def _binary_op(self, other, op):
        return op(np.asarray(self.value), other)

    __mul__ = lambda self, other: self._binary_op(other, np.multiply)
    __rmul__ = __mul__
    __add__ = lambda self, other: self._binary_op(other, np.add)
    __radd__ = __add__
    __sub__ = lambda self, other: self._binary_op(other, np.subtract)
    __rsub__ = lambda self, other: np.subtract(other, np.asarray(self.value))
    __truediv__ = lambda self, other: self._binary_op(other, np.divide)
    __rtruediv__ = lambda self, other: np.divide(other, np.asarray(self.value))
    __neg__ = lambda self: np.negative(np.asarray(self.value))


def source_full(
    r,
    p0,
    rho0,
    drho0,
    d2rho0,
    h00,
    dh00,
    lamb,
    nu,
    eta,
    deta,
    d2eta,
):
    """
    Evaluate the full (non-polynomial) L=2 source term for h00.
    Inputs can be scalars or NumPy arrays of matching shape.
    """
    fn = _get_full_source_fn()

    def _wrap(x):
        return x if callable(x) else _CallableArray(x)

    eta_obj = _wrap(eta)
    deta_obj = _wrap(deta)
    d2eta_obj = _wrap(d2eta)

    return fn(r, p0, rho0, drho0, d2rho0, h00, dh00, lamb, nu, eta_obj, deta_obj, d2eta_obj)


def source_r3(r, K, pc, a0=1.0, nu_c=0.0, T=1.0):
    """
    Analytic series source term up to r^3.
    """
    fn = _get_r3_fn()
    return fn(r, K, pc, a0, nu_c, T)


def source_r5(r, K, pc, a0=1.0, nu_c=0.0, T=1.0):
    """
    Analytic series source term up to r^5.
    """
    fn = _get_r5_fn()
    return fn(r, K, pc, a0, nu_c, T)


def source_r7(r, K, pc, a0=1.0, nu_c=0.0, T=1.0):
    """
    Analytic series source term up to r^7.
    """
    fn = _get_r7_fn()
    return fn(r, K, pc, a0, nu_c, T)

def source_r9(r, K, pc, a0=1.0, nu_c=0.0, T=1.0):
    """
    Analytic series source term up to r^9.
    """
    fn = _get_r9_fn()
    return fn(r, K, pc, a0, nu_c, T)


def source_series(r, K, pc, a0=1.0, nu_c=0.0, T=1.0, order=3):
    """
    Convenience dispatcher for analytic series source terms.
    order can be 3, 5, or 7.
    """
    if order == 3:
        return source_r3(r, K, pc, a0=a0, nu_c=nu_c, T=T)
    if order == 5:
        return source_r5(r, K, pc, a0=a0, nu_c=nu_c, T=T)
    if order == 7:
        return source_r7(r, K, pc, a0=a0, nu_c=nu_c, T=T)
    if order == 9:
        return source_r9(r, K, pc, a0=a0, nu_c=nu_c, T=T)
    
    raise ValueError(f"Unsupported order={order}, expected 3, 5, 7, or 9.")
