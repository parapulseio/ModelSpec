"""Tokenizer extractor — chat template, type and vocab size."""

from __future__ import annotations

import json
from pathlib import Path

from modelspec.extractors.base import ExtractionSource
from modelspec.extractors.tokenizer import TokenizerExtractor


def _claims(tmp_path: Path, files: dict[str, dict]):
    repo_files = []
    for name, payload in files.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
        repo_files.append(name)
    src = ExtractionSource(root=tmp_path, repo_files=repo_files)
    result = TokenizerExtractor().extract(src)
    return {c.field_path: c.value for c in result.claims}, result


def test_chat_template_and_type_and_vocab(tmp_path: Path):
    claims, _ = _claims(
        tmp_path,
        {
            "tokenizer_config.json": {
                "chat_template": "{{ messages }}",
                "tokenizer_class": "PreTrainedTokenizerFast",
            },
            "tokenizer.json": {
                "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2}},
                "added_tokens": [{"id": 3, "content": "<eos>"}],
            },
        },
    )
    assert claims["tokenizer.chat_template_present"] is True
    assert claims["tokenizer.type"] == "BPE"
    assert claims["tokenizer.vocab_size"] == 4  # 3 vocab + 1 added


def test_no_chat_template(tmp_path: Path):
    claims, _ = _claims(tmp_path, {"tokenizer_config.json": {"tokenizer_class": "X"}})
    assert claims["tokenizer.chat_template_present"] is False
