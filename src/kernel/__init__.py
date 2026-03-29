"""CiteWeave kernel layer.

This package exposes stable, adapter-agnostic application services.
CLI, OpenClaw, future HTTP APIs, and other entrypoints should depend on this
layer instead of reaching directly into scattered processing/agent modules.
"""

from .service import CiteWeaveKernel
from .batch_tracker import BatchUploadTracker

__all__ = ["CiteWeaveKernel", "BatchUploadTracker"]
