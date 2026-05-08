"""Shared pytest safeguards for import-stubbing tests."""

from __future__ import annotations

import sys


_TRANSIENT_STUB_MODULES = {
    "prompt_toolkit",
    "src",
    "src.processing",
    "src.processing.pdf",
    "src.processing.pdf.document_processor",
    "src.agents",
    "src.agents.multi_agent_research_system",
    "src.agents.routing",
    "src.kernel",
    "src.kernel.batch_tracker",
    "src.kernel.query_history",
}


def _is_unrestored_stub(module) -> bool:
    """Return True for hand-built ModuleType stubs left in sys.modules."""
    return getattr(module, "__spec__", None) is None


def pytest_runtest_teardown(item, nextitem):
    leaked = sorted(
        name
        for name in _TRANSIENT_STUB_MODULES
        if name in sys.modules and _is_unrestored_stub(sys.modules[name])
    )
    assert not leaked, f"temporary module stubs leaked across tests: {', '.join(leaked)}"
