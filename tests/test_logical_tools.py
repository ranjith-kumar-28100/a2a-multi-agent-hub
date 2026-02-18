"""Tests for the Logical Reasoning Agent's tools.

These tests validate the core logic functions directly,
without requiring an LLM or A2A server.
"""

import pytest

from agents.logical_reasoning.agent import (
    logical_and,
    logical_or,
    logical_not,
    logical_xor,
    logical_implies,
    logical_biconditional,
    evaluate_expression,
)


# ─── AND ─────────────────────────────────────────────────────────────


class TestLogicalAnd:
    """Tests for the logical AND operation."""

    def test_true_and_true(self):
        assert logical_and(True, True) is True

    def test_true_and_false(self):
        assert logical_and(True, False) is False

    def test_false_and_true(self):
        assert logical_and(False, True) is False

    def test_false_and_false(self):
        assert logical_and(False, False) is False


# ─── OR ──────────────────────────────────────────────────────────────


class TestLogicalOr:
    """Tests for the logical OR operation."""

    def test_true_or_true(self):
        assert logical_or(True, True) is True

    def test_true_or_false(self):
        assert logical_or(True, False) is True

    def test_false_or_true(self):
        assert logical_or(False, True) is True

    def test_false_or_false(self):
        assert logical_or(False, False) is False


# ─── NOT ─────────────────────────────────────────────────────────────


class TestLogicalNot:
    """Tests for the logical NOT operation."""

    def test_not_true(self):
        assert logical_not(True) is False

    def test_not_false(self):
        assert logical_not(False) is True


# ─── XOR ─────────────────────────────────────────────────────────────


class TestLogicalXor:
    """Tests for the logical XOR operation."""

    def test_true_xor_true(self):
        assert logical_xor(True, True) is False

    def test_true_xor_false(self):
        assert logical_xor(True, False) is True

    def test_false_xor_true(self):
        assert logical_xor(False, True) is True

    def test_false_xor_false(self):
        assert logical_xor(False, False) is False


# ─── IMPLIES ─────────────────────────────────────────────────────────


class TestLogicalImplies:
    """Tests for the logical IMPLIES (p → q) operation."""

    def test_true_implies_true(self):
        assert logical_implies(True, True) is True

    def test_true_implies_false(self):
        # Only case where implication is False
        assert logical_implies(True, False) is False

    def test_false_implies_true(self):
        assert logical_implies(False, True) is True

    def test_false_implies_false(self):
        assert logical_implies(False, False) is True


# ─── BICONDITIONAL ───────────────────────────────────────────────────


class TestLogicalBiconditional:
    """Tests for the logical BICONDITIONAL (p ↔ q) operation."""

    def test_true_biconditional_true(self):
        assert logical_biconditional(True, True) is True

    def test_true_biconditional_false(self):
        assert logical_biconditional(True, False) is False

    def test_false_biconditional_true(self):
        assert logical_biconditional(False, True) is False

    def test_false_biconditional_false(self):
        assert logical_biconditional(False, False) is True


# ─── Expression Evaluator ────────────────────────────────────────────


class TestEvaluateExpression:
    """Tests for the logical expression evaluator."""

    def test_simple_and(self):
        assert evaluate_expression("True and False") == "False"

    def test_simple_or(self):
        assert evaluate_expression("True or False") == "True"

    def test_simple_not(self):
        assert evaluate_expression("not True") == "False"

    def test_parenthesized_expression(self):
        assert evaluate_expression("True and (False or True)") == "True"

    def test_complex_expression(self):
        assert evaluate_expression("not (True and False)") == "True"

    def test_nested_parentheses(self):
        assert evaluate_expression("(True or False) and (not False)") == "True"

    def test_de_morgans_law(self):
        # NOT (A AND B) == (NOT A) OR (NOT B)
        result1 = evaluate_expression("not (True and False)")
        result2 = evaluate_expression("(not True) or (not False)")
        assert result1 == result2

    def test_invalid_token_returns_error(self):
        result = evaluate_expression("True and maybe")
        assert "Error" in result

    def test_syntax_error_returns_error(self):
        result = evaluate_expression("True and and False")
        assert "Error" in result

    def test_all_true(self):
        assert evaluate_expression("True and True and True") == "True"

    def test_all_false(self):
        assert evaluate_expression("False or False or False") == "False"
