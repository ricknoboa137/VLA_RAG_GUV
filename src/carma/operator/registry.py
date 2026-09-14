"""Operator interface registry."""

from __future__ import annotations

from carma.registry import Registry
from carma.types import OperatorInterface

OPERATORS: Registry[OperatorInterface] = Registry("operator")
