"""Tokenizer extractor — reads tokenizer_config.json / tokenizer.json / generation_config.json.

Emits tokenizer type, vocab size, chat-template presence, and special-token ids.
Claims use the ``tokenizer`` source label (distinct from config.json's ``config``)
so multi-source conflicts — e.g. config vs tokenizer vocab_size — are legible in
the coverage conflict histogram. tokenizer.json can be large but it is metadata
(already fetched), and we only read a few keys. See docs/extractors.md.
"""

from __future__ import annotations

import json

from modelspec.extractors.base import ExtractionSource, ExtractorResult, FieldClaim

# tokenizers model.type -> normalized tokenizer type label.
_TYPE_MAP = {
    "BPE": "BPE",
    "Unigram": "Unigram",
    "WordPiece": "WordPiece",
    "WordLevel": "WordLevel",
}

_SRC = "tokenizer"


class TokenizerExtractor:
    name = "tokenizer"

    def can_handle(self, source: ExtractionSource) -> bool:
        return (
            source.has("tokenizer_config.json")
            or source.has("tokenizer.json")
            or source.has("generation_config.json")
        )

    def extract(self, source: ExtractionSource) -> ExtractorResult:
        claims: list[FieldClaim] = []
        passthrough: dict = {}

        # tokenizer_config.json: chat template + class hint.
        if source.has("tokenizer_config.json"):
            cfg = json.loads(
                source.path("tokenizer_config.json").read_text(encoding="utf-8")
            )
            has_template = bool(cfg.get("chat_template"))
            claims.append(
                FieldClaim("tokenizer.chat_template_present", has_template, _SRC, "high")
            )
            if "model_max_length" in cfg:
                passthrough["model_max_length"] = cfg["model_max_length"]
            tok_class = cfg.get("tokenizer_class")
            if tok_class:
                passthrough["tokenizer_class"] = tok_class

        # tokenizer.json: authoritative type + vocab size (wins over config's
        # often-padded vocab_size, which is emitted at "medium").
        if source.has("tokenizer.json"):
            tok = json.loads(source.path("tokenizer.json").read_text(encoding="utf-8"))
            model = tok.get("model") or {}
            mtype = model.get("type")
            if mtype:
                claims.append(
                    FieldClaim("tokenizer.type", _TYPE_MAP.get(mtype, mtype), _SRC, "high")
                )
            vocab = model.get("vocab")
            added = tok.get("added_tokens") or []
            if isinstance(vocab, (dict, list)):  # Unigram stores vocab as a list
                claims.append(
                    FieldClaim("tokenizer.vocab_size", len(vocab) + len(added), _SRC, "high")
                )

        # generation_config.json: special-token ids as a fallback when config.json
        # omits them (emitted at "medium" so config.json wins when it has them).
        if source.has("generation_config.json"):
            gen = json.loads(
                source.path("generation_config.json").read_text(encoding="utf-8")
            )
            for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
                if gen.get(key) is not None:
                    claims.append(FieldClaim(f"tokenizer.{key}", gen[key], _SRC, "medium"))

        return ExtractorResult(claims=claims, passthrough=passthrough)
