"""ParaPulse ModelSpec — extract and normalize LLM model specifications.

Public entry points:
    - ``modelspec.schema.ModelSpec``: the unified Pydantic v2 schema.
    - ``modelspec.pipeline.extract``: run the end-to-end extraction pipeline.
"""

from modelspec.schema import ModelSpec

__version__ = "0.1.0"

__all__ = ["ModelSpec", "__version__"]
