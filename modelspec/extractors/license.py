"""License extractor — tiered identification (see docs/extractors.md).

Tier 1 (fingerprint): normalize the license text (collapse whitespace,
lowercase) and match against marker-phrase signatures of common licenses. We use
marker phrases rather than full-text hashes because model LICENSE files vary in
surrounding boilerplate while their identifying phrases are stable.

Tier 2 (keyword): for unmatched text, scan capability keywords to emit
capability flags (commercial_use / redistribution / attribution_required).

Tier 3 (LLM): a low-confidence fallback classifier — out of scope here; left as
a hook (see --no-license-llm in docs/cli.md). M2 stops at tier 2.

Note: never match only ``LICENSE*`` — MODEL_LICENSE / USE_POLICY.md / Notice are
real names too. HF front-matter ``license:`` is auxiliary evidence, not ground
truth.
"""

from __future__ import annotations

import re

from modelspec.extractors.base import ExtractionSource, ExtractorResult, FieldClaim

# Candidate license file names, in priority order.
_LICENSE_FILES = [
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "MODEL_LICENSE",
    "MODEL_LICENSE.md",
    "USE_POLICY.md",
    "Notice",
    "NOTICE",
]

# Tier 1 signatures: spdx_id -> all marker phrases that must be present
# (matched against normalized lowercase text).
_SIGNATURES: dict[str, list[str]] = {
    "apache-2.0": ["apache license", "version 2.0"],
    "mit": ["mit license"],
    "bsd-3-clause": ["redistribution and use in source", "neither the name"],
    "gpl-3.0": ["gnu general public license", "version 3"],
    "agpl-3.0": ["gnu affero general public license"],
    "cc-by-4.0": ["creative commons attribution 4.0"],
    "cc-by-nc-4.0": ["attribution-noncommercial 4.0"],
    "cc-by-sa-4.0": ["attribution-sharealike 4.0"],
    "llama2": ["llama 2 community license"],
    "llama3": ["llama 3 community license"],
    "llama3.1": ["llama 3.1 community license"],
    "gemma": ["gemma terms of use"],
    "qwen": ["qwen license agreement"],
    "openrail": ["openrail"],
    "openrail-m": ["bigscience openrail-m"],
    "apple-ascl": ["apple sample code license"],
}

# Known capability flags per SPDX id (tier-1 confidence). Permissive OSI
# licenses allow commercial use + redistribution and require attribution.
_FLAGS: dict[str, dict[str, bool]] = {
    "apache-2.0": {"commercial_use": True, "redistribution": True, "attribution_required": True},
    "mit": {"commercial_use": True, "redistribution": True, "attribution_required": True},
    "bsd-3-clause": {"commercial_use": True, "redistribution": True, "attribution_required": True},
    "gpl-3.0": {"commercial_use": True, "redistribution": True, "attribution_required": True},
    "agpl-3.0": {"commercial_use": True, "redistribution": True, "attribution_required": True},
    "cc-by-4.0": {"commercial_use": True, "redistribution": True, "attribution_required": True},
    "cc-by-nc-4.0": {"commercial_use": False, "redistribution": True, "attribution_required": True},
    "cc-by-sa-4.0": {"commercial_use": True, "redistribution": True, "attribution_required": True},
    "llama2": {"commercial_use": True, "redistribution": True, "attribution_required": True},
    "llama3": {"commercial_use": True, "redistribution": True, "attribution_required": True},
    "llama3.1": {"commercial_use": True, "redistribution": True, "attribution_required": True},
    "gemma": {"commercial_use": True, "redistribution": True, "attribution_required": True},
    "qwen": {"commercial_use": True, "redistribution": True, "attribution_required": True},
    "openrail": {"commercial_use": True, "redistribution": True, "attribution_required": False},
    "openrail-m": {"commercial_use": True, "redistribution": True, "attribution_required": False},
}

def _normalize(text: str) -> str:
    """Lowercase and collapse all whitespace runs to single spaces."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _front_matter_license(text: str) -> str | None:
    """Pull ``license:`` out of a README YAML front-matter block (aux evidence)."""
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    block = text[3:end] if end != -1 else text[3:]
    m = re.search(r"^\s*license:\s*([^\s#]+)", block, re.MULTILINE)
    return m.group(1).strip().strip("\"'") if m else None


class LicenseExtractor:
    name = "license"

    def can_handle(self, source: ExtractionSource) -> bool:
        if any(source.has(n) for n in _LICENSE_FILES):
            return True
        return source.has("README.md")

    def _read_license_text(self, source: ExtractionSource) -> str | None:
        for name in _LICENSE_FILES:
            if source.has(name):
                return source.path(name).read_text(encoding="utf-8", errors="replace")
        return None

    def extract(self, source: ExtractionSource) -> ExtractorResult:
        claims: list[FieldClaim] = []
        text = self._read_license_text(source)

        if text is not None:
            norm = _normalize(text)

            # --- Tier 1: signature match ---
            spdx = None
            for candidate, markers in _SIGNATURES.items():
                if all(m in norm for m in markers):
                    spdx = candidate
                    break

            if spdx is not None:
                claims.append(FieldClaim("license.spdx_id", spdx, "fingerprint", "high"))
                claims.append(
                    FieldClaim("license.confidence_tier", "fingerprint", "fingerprint", "high")
                )
                for flag, value in _FLAGS.get(spdx, {}).items():
                    claims.append(FieldClaim(f"license.{flag}", value, "fingerprint", "high"))
            else:
                # --- Tier 2: capability keywords ---
                claims.append(
                    FieldClaim("license.confidence_tier", "keyword", "keyword", "low")
                )
                if re.search(r"non-?commercial", norm):
                    claims.append(FieldClaim("license.commercial_use", False, "keyword", "low"))
                elif "commercial use" in norm:
                    claims.append(FieldClaim("license.commercial_use", True, "keyword", "low"))
                if re.search(r"redistribut", norm):
                    claims.append(FieldClaim("license.redistribution", True, "keyword", "low"))
                if "attribution" in norm:
                    claims.append(
                        FieldClaim("license.attribution_required", True, "keyword", "low")
                    )

        # --- Auxiliary: README front-matter license (low confidence) ---
        if source.has("README.md"):
            fm = _front_matter_license(
                source.path("README.md").read_text(encoding="utf-8", errors="replace")
            )
            if fm:
                # Lower confidence than file fingerprint, so it loses on conflict
                # but fills the field when no LICENSE file exists.
                claims.append(FieldClaim("license.spdx_id", fm, "heuristic", "low"))

        return ExtractorResult(claims=claims)
