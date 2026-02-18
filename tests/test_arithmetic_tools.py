"""Tests for the Arithmetic Agent's tools.

These tests validate the core arithmetic functions directly,
without requiring an LLM or A2A server.
"""

import pytest

from agents.arithmetic.agent import add, subtract, multiply, divide, modulo, power


# ─── Addition ────────────────────────────────────────────────────────


class TestAdditionTool:
    """Tests for the AdditionTool."""

    def test_add_positive_numbers(self):
        assert add.func(3, 5) == 8

    def test_add_negative_numbers(self):
        assert add.func(-3, -5) == -8

    def test_add_mixed_sign(self):
        assert add.func(-3, 5) == 2

    def test_add_zero(self):
        assert add.func(0, 5) == 5

    def test_add_floats(self):
        assert add.func(1.5, 2.3) == pytest.approx(3.8)

    def test_add_large_numbers(self):
        assert add.func(1e10, 2e10) == 3e10


# ─── Subtraction ─────────────────────────────────────────────────────


class TestSubtractionTool:
    """Tests for the SubtractionTool."""

    def test_subtract_positive(self):
        assert subtract.func(10, 3) == 7

    def test_subtract_negative_result(self):
        assert subtract.func(3, 10) == -7

    def test_subtract_zero(self):
        assert subtract.func(5, 0) == 5

    def test_subtract_from_zero(self):
        assert subtract.func(0, 5) == -5

    def test_subtract_same_numbers(self):
        assert subtract.func(7, 7) == 0


# ─── Multiplication ──────────────────────────────────────────────────


class TestMultiplicationTool:
    """Tests for the MultiplicationTool."""

    def test_multiply_positive(self):
        assert multiply.func(4, 5) == 20

    def test_multiply_by_zero(self):
        assert multiply.func(100, 0) == 0

    def test_multiply_negative(self):
        assert multiply.func(-3, 4) == -12

    def test_multiply_two_negatives(self):
        assert multiply.func(-3, -4) == 12

    def test_multiply_floats(self):
        assert multiply.func(2.5, 4) == pytest.approx(10.0)

    def test_multiply_by_one(self):
        assert multiply.func(42, 1) == 42


# ─── Division ────────────────────────────────────────────────────────


class TestDivisionTool:
    """Tests for the DivisionTool."""

    def test_divide_evenly(self):
        assert divide.func(10, 2) == 5.0

    def test_divide_with_remainder(self):
        assert divide.func(7, 2) == 3.5

    def test_divide_by_one(self):
        assert divide.func(42, 1) == 42.0

    def test_divide_negative(self):
        assert divide.func(-10, 2) == -5.0

    def test_divide_by_zero_raises(self):
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide.func(10, 0)

    def test_divide_zero_by_number(self):
        assert divide.func(0, 5) == 0.0


# ─── Modulo ──────────────────────────────────────────────────────────


class TestModuloTool:
    """Tests for the ModuloTool."""

    def test_modulo_with_remainder(self):
        assert modulo.func(17, 5) == 2

    def test_modulo_no_remainder(self):
        assert modulo.func(10, 5) == 0

    def test_modulo_smaller_dividend(self):
        assert modulo.func(3, 5) == 3

    def test_modulo_by_zero_raises(self):
        with pytest.raises(ValueError, match="Cannot modulo by zero"):
            modulo.func(10, 0)

    def test_modulo_one(self):
        assert modulo.func(100, 1) == 0


# ─── Power ───────────────────────────────────────────────────────────


class TestPowerTool:
    """Tests for the PowerTool."""

    def test_power_positive(self):
        assert power.func(2, 10) == 1024

    def test_power_zero_exponent(self):
        assert power.func(5, 0) == 1

    def test_power_one_exponent(self):
        assert power.func(5, 1) == 5

    def test_power_square(self):
        assert power.func(3, 2) == 9

    def test_power_negative_exponent(self):
        assert power.func(2, -1) == pytest.approx(0.5)

    def test_power_zero_base(self):
        assert power.func(0, 5) == 0

    def test_power_fractional_exponent(self):
        assert power.func(4, 0.5) == pytest.approx(2.0)
