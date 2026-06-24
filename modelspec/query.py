"""Composable predicates for filtering collections of ``ModelSpec`` (M5).

The per-spec accessors (``spec.is_quantized()``, ``spec.effective_context`` …)
live on the model itself. This module is the *collection* layer: small predicate
factories that return ``Predicate`` callables, plus combinators, so downstream
consumers can express corpus queries declaratively::

    from modelspec.query import filter_specs, is_quantized, min_params, all_of

    big_quants = filter_specs(specs, all_of(is_quantized, min_params(7e9)))

Predicates are plain ``Callable[[ModelSpec], bool]`` — any function with that
shape composes here, no base class required.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator

from modelspec.schema import ModelSpec

Predicate = Callable[[ModelSpec], bool]


# --- Bare predicates (use directly, no call needed) ----------------------- #


def is_quantized(spec: ModelSpec) -> bool:
    return spec.is_quantized()


def is_merged(spec: ModelSpec) -> bool:
    return spec.is_merged()


def is_moe(spec: ModelSpec) -> bool:
    return spec.is_moe()


def is_dense(spec: ModelSpec) -> bool:
    return not spec.is_moe()


def is_adapter(spec: ModelSpec) -> bool:
    return spec.is_adapter()


def is_derived(spec: ModelSpec) -> bool:
    return spec.is_derived()


def is_decoder_only(spec: ModelSpec) -> bool:
    return spec.is_decoder_only()


def is_multimodal(spec: ModelSpec) -> bool:
    return spec.is_multimodal()


def has_chat_template(spec: ModelSpec) -> bool:
    """True if the tokenizer ships a chat template (a weak instruct-model proxy)."""
    return spec.tokenizer.chat_template_present is True


def commercial_use_allowed(spec: ModelSpec) -> bool:
    """Strict: only True when the license explicitly permits commercial use."""
    return spec.license.commercial_use is True


# --- Predicate factories (call to bind a parameter) ----------------------- #


def family_is(*families: str) -> Predicate:
    """Match an architecture family (case-insensitive), e.g. ``family_is("llama", "qwen2")``."""
    wanted = {f.lower() for f in families}
    return lambda spec: (spec.architecture.family or "").lower() in wanted


def quant_format_in(*formats: str) -> Predicate:
    """Match a quantization format discriminator, e.g. ``quant_format_in("gguf", "awq")``."""
    wanted = {f.lower() for f in formats}
    return lambda spec: spec.quant_format is not None and spec.quant_format.lower() in wanted


def modality_is(*modalities: str) -> Predicate:
    """Match the coarse modality, e.g. ``modality_is("decoder-only", "multimodal")``.

    Useful to keep non-decoder models (audio / vision / encoder / seq2seq) out of
    an LLM-focused query.
    """
    wanted = {m.lower() for m in modalities}
    return lambda spec: spec.modality.lower() in wanted


def min_params(n: float) -> Predicate:
    """Total parameter count >= ``n`` (unknown counts are excluded)."""
    return lambda spec: spec.parameters.total is not None and spec.parameters.total >= n


def max_params(n: float) -> Predicate:
    """Total parameter count <= ``n`` (unknown counts are excluded)."""
    return lambda spec: spec.parameters.total is not None and spec.parameters.total <= n


def min_context(n: int) -> Predicate:
    """Effective context window >= ``n`` (unknown windows are excluded)."""
    return lambda spec: (spec.effective_context or 0) >= n


def license_is(*spdx_ids: str) -> Predicate:
    """Match a license SPDX id (case-insensitive)."""
    wanted = {s.lower() for s in spdx_ids}
    return lambda spec: (spec.license.spdx_id or "").lower() in wanted


# --- Combinators ---------------------------------------------------------- #


def all_of(*predicates: Predicate) -> Predicate:
    """Logical AND of every predicate (vacuously True with no args)."""
    return lambda spec: all(p(spec) for p in predicates)


def any_of(*predicates: Predicate) -> Predicate:
    """Logical OR of every predicate (vacuously False with no args)."""
    return lambda spec: any(p(spec) for p in predicates)


def negate(predicate: Predicate) -> Predicate:
    """Logical NOT of a predicate."""
    return lambda spec: not predicate(spec)


# --- Driver --------------------------------------------------------------- #


def filter_specs(specs: Iterable[ModelSpec], *predicates: Predicate) -> Iterator[ModelSpec]:
    """Yield specs satisfying *all* given predicates (implicit AND)."""
    combined = all_of(*predicates)
    return (s for s in specs if combined(s))
