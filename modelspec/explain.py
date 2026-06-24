"""Self-documenting field catalog for ``ModelSpec`` (M5).

Every field in the schema already carries a ``description=`` (see
``schema/spec.py``), so we don't keep a second copy of the docs — we introspect
the live Pydantic models and flatten them into dotted-path ``FieldDoc`` entries.
This drives ``modelspec explain <field>`` and is reusable by any consumer that
wants per-field help (UI tooltips, generated forms).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union, get_args, get_origin

from pydantic import BaseModel

from modelspec.schema.spec import ModelSpec


@dataclass(frozen=True)
class FieldDoc:
    """One documented field, addressed by its dotted path."""

    path: str
    type_str: str
    description: str
    choices: tuple[str, ...] = field(default_factory=tuple)


def _strip_annotated(tp):
    # Annotated[X, ...] (e.g. the Quantization discriminated union) exposes the
    # underlying type as __origin__ and the extras as __metadata__.
    if hasattr(tp, "__metadata__"):
        return tp.__origin__
    return tp


def _is_model(tp) -> bool:
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def _type_name(tp) -> str:
    tp = _strip_annotated(tp)
    origin = get_origin(tp)
    if tp is type(None):
        return "null"
    if origin is None:
        if _is_model(tp):
            return tp.__name__
        return getattr(tp, "__name__", str(tp))
    if origin is Literal:
        return " | ".join(repr(a) for a in get_args(tp))
    if origin is list:
        return f"list[{_type_name(get_args(tp)[0])}]"
    if origin is dict:
        k, v = get_args(tp)
        return f"dict[{_type_name(k)}, {_type_name(v)}]"
    if origin is Union:
        # Render Optional[X] as "X" — the None-ness is implied by the schema.
        args = [a for a in get_args(tp) if a is not type(None)]
        return " | ".join(_type_name(a) for a in args)
    return getattr(tp, "__name__", str(tp))


def _choices(tp) -> tuple[str, ...]:
    """Literal enum values reachable inside the annotation (unwrapping Optional)."""
    tp = _strip_annotated(tp)
    origin = get_origin(tp)
    if origin is Literal:
        return tuple(str(a) for a in get_args(tp))
    if origin is Union:
        for a in get_args(tp):
            found = _choices(a)
            if found:
                return found
    return ()


def _nested_models(tp) -> list[type[BaseModel]]:
    """BaseModel classes reachable through Optional / Union / list / dict-value."""
    tp = _strip_annotated(tp)
    if _is_model(tp):
        return [tp]
    origin = get_origin(tp)
    if origin is None or origin is Literal:
        return []
    out: list[type[BaseModel]] = []
    args = get_args(tp)
    if origin is dict:
        args = args[1:]  # only the value type can hold a model
    for a in args:
        out.extend(_nested_models(a))
    return out


def field_catalog() -> list[FieldDoc]:
    """Every documented field of ``ModelSpec``, in stable depth-first order."""
    docs: dict[str, FieldDoc] = {}
    _collect(ModelSpec, "", docs)
    return list(docs.values())


def _collect(model: type[BaseModel], prefix: str, docs: dict[str, FieldDoc]) -> None:
    for name, fld in model.model_fields.items():
        path = f"{prefix}{name}"
        if path in docs:  # union members can share a sub-path (e.g. quant format)
            continue
        ann = fld.annotation
        docs[path] = FieldDoc(
            path=path,
            type_str=_type_name(ann),
            description=fld.description or "",
            choices=_choices(ann),
        )
        for sub in _nested_models(ann):
            _collect(sub, path + ".", docs)


def explain_field(query: str) -> list[FieldDoc]:
    """Look up a field by exact dotted path, then by suffix, then by substring.

    The progressively looser matching lets callers type ``family`` or
    ``architecture.family`` and still land on the right field.
    """
    catalog = field_catalog()
    exact = [d for d in catalog if d.path == query]
    if exact:
        return exact
    by_leaf = [d for d in catalog if d.path.split(".")[-1] == query]
    if by_leaf:
        return by_leaf
    return [d for d in catalog if query.lower() in d.path.lower()]
