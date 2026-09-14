"""Error hierarchy.

Every failure raised inside ``carma`` derives from :class:`CarmaError`. Public
functions signal failure by raising, never by returning ``None``.
"""

from __future__ import annotations


class CarmaError(Exception):
    """Base class for every error raised by this project."""


class ConfigError(CarmaError):
    """A configuration file is malformed, incomplete, or names an unknown key."""


class MemoryError_(CarmaError):
    """A memory store operation failed.

    Named with a trailing underscore to avoid shadowing the builtin.
    """


class SimulatorError(CarmaError):
    """The simulator could not be reached, reset, or stepped."""


class OperatorUnavailableError(CarmaError):
    """A query was issued but no operator channel is connected."""


class AnalyzerError(CarmaError):
    """A scene analyzer could not be loaded or could not process an image."""


class DeterminismError(CarmaError):
    """A component behaved non-deterministically under a fixed seed."""
