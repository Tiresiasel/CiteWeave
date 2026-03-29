"""Adapter layer for CiteWeave.

Adapters translate external entrypoints into calls on the kernel layer.
Examples: CLI, OpenClaw Skill, future HTTP API.
"""

from .openclaw_facade import OpenClawCiteWeaveFacade

__all__ = ["OpenClawCiteWeaveFacade"]
