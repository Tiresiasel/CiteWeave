"""Top-level package for CiteWeave.

Keep package import side effects minimal.
Heavy runtime objects (PDF processors, agent systems) should be imported from
specific modules or through the kernel/adapters layer, not from `src` itself.
"""

__all__ = []
