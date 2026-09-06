"""ParaPulse ModelSpec — extract and normalize LLM model specifications.

Public entry points:
    - ``modelspec.schema.ModelSpec``: the unified Pydantic v2 schema, with
      convenience accessors (``is_quantized()``, ``effective_context``, …).
    - ``modelspec.pipeline.extract``: run the end-to-end extraction pipeline.
    - ``modelspec.query``: composable predicates for filtering spec collections.
    - ``modelspec.explain``: per-field documentation (``field_catalog`` /
      ``explain_field``).
"""

from importlib.metadata import PackageNotFoundError, version

from modelspec.schema import ModelSpec

try:
    __version__ = version("modelspec")
except PackageNotFoundError:  # pragma: no cover - running from source, not installed
    __version__ = "0.0.0+unknown"

__all__ = ["ModelSpec", "__version__"]
