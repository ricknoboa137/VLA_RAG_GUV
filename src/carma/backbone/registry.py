"""Backbone registry."""

from __future__ import annotations

from carma.registry import Registry
from carma.types import NavigationBackbone

BACKBONES: Registry[NavigationBackbone] = Registry("backbone")
