"""ParaPulse ModelSpec — extract and normalize LLM model specifications.

Public entry points:
    - ``modelspec.schema.ModelSpec``: the unified Pydantic v2 schema, with
      convenience accessors (``is_quantized()``, ``effective_context``, …).
    - ``modelspec.pipeline.extract``: run the end-to-end extraction pipeline.
    - ``modelspec.query``: composable predicates for filtering spec collections.
    - ``modelspec.explain``: per-field documentation (``field_catalog`` /
      ``explain_field``).
"""

from modelspec.schema import ModelSpec

__version__ = "0.1.0"

__all__ = ["ModelSpec", "__version__"]
