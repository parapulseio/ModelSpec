"""License extractor — fingerprint, keyword fallback, and front-matter aux."""

from __future__ import annotations

from pathlib import Path

from modelspec.extractors.base import ExtractionSource
from modelspec.extractors.license import LicenseExtractor

APACHE = """
                                 Apache License
                           Version 2.0, January 2004

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
"""

MIT = """MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software ... The above copyright notice shall be included.
"""


def _claims(tmp_path: Path, files: dict[str, str]):
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    src = ExtractionSource(root=tmp_path, repo_files=list(files))
    result = LicenseExtractor().extract(src)
    return {c.field_path: (c.value, c.source) for c in result.claims}


def test_apache_fingerprint(tmp_path: Path):
    claims = _claims(tmp_path, {"LICENSE": APACHE})
    assert claims["license.spdx_id"][0] == "apache-2.0"
    assert claims["license.confidence_tier"][0] == "fingerprint"
    assert claims["license.commercial_use"][0] is True


def test_mit_fingerprint(tmp_path: Path):
    claims = _claims(tmp_path, {"LICENSE.md": MIT})
    assert claims["license.spdx_id"][0] == "mit"


def test_non_license_filename_still_scanned(tmp_path: Path):
    # MODEL_LICENSE must be picked up, not just LICENSE*.
    claims = _claims(tmp_path, {"MODEL_LICENSE": "Gemma Terms of Use\n..."})
    assert claims["license.spdx_id"][0] == "gemma"


def test_keyword_fallback_for_unknown_text(tmp_path: Path):
    text = "This is a custom non-commercial license. You may not redistribute."
    claims = _claims(tmp_path, {"LICENSE": text})
    assert "license.spdx_id" not in claims  # no signature matched
    assert claims["license.confidence_tier"][0] == "keyword"
    assert claims["license.commercial_use"][0] is False


def test_readme_front_matter_is_aux_low_confidence(tmp_path: Path):
    readme = "---\nlicense: apache-2.0\ntags:\n- text-generation\n---\n# Model\n"
    claims = _claims(tmp_path, {"README.md": readme})
    assert claims["license.spdx_id"] == ("apache-2.0", "heuristic")
