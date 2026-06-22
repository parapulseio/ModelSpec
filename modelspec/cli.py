"""Command-line interface: ``modelspec extract <repo_id|path>``.

See docs/cli.md for the full option reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from modelspec import __version__
from modelspec.pipeline import extract
from modelspec.schema import ModelSpec


def _to_output_dict(spec: ModelSpec, show_provenance: bool) -> dict[str, Any]:
    data = spec.model_dump(mode="json")
    if not show_provenance:
        # Keep the default output concise: drop heavy / archival provenance,
        # but keep the actionable bits (conflicts, warnings, unknown_fields).
        prov = data.get("provenance", {})
        prov.pop("per_field", None)
        prov.pop("raw_config_json", None)
        prov.pop("raw_gguf_kv", None)
    return data


def _render(data: dict[str, Any], fmt: str) -> str:
    if fmt == "yaml":
        try:
            import yaml
        except ImportError:
            print(
                "error: --format yaml requires PyYAML (pip install 'modelspec[yaml]')",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    return json.dumps(data, indent=2, ensure_ascii=False)


def _cmd_extract(args: argparse.Namespace) -> int:
    try:
        spec = extract(args.repo_id, revision=args.revision, offline=args.offline)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # network / parse / validation
        print(f"error: extraction failed: {e}", file=sys.stderr)
        return 1

    if args.strict and spec.provenance.warnings:
        for w in spec.provenance.warnings:
            print(f"warning: {w}", file=sys.stderr)
        return 1

    out = _render(_to_output_dict(spec, args.show_provenance), args.format)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
            if not out.endswith("\n"):
                f.write("\n")
    else:
        print(out)
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    print(json.dumps(ModelSpec.model_json_schema(), indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modelspec",
        description="Extract and normalize LLM model specifications.",
    )
    parser.add_argument("--version", action="version", version=f"modelspec {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="extract a ModelSpec from a repo id or local dir")
    p_extract.add_argument("repo_id", help="HF repo id (e.g. meta-llama/Llama-3.1-8B) or local path")
    p_extract.add_argument("--format", choices=["json", "yaml"], default="json")
    p_extract.add_argument("-o", "--output", help="write to file instead of stdout")
    p_extract.add_argument("--offline", action="store_true", help="local paths only, no network")
    p_extract.add_argument("--revision", help="commit / branch / tag")
    p_extract.add_argument(
        "--show-provenance", action="store_true", help="include full provenance (per_field, raw)"
    )
    p_extract.add_argument(
        "--strict", action="store_true", help="non-zero exit if any cross-field warning fires"
    )
    p_extract.set_defaults(func=_cmd_extract)

    p_schema = sub.add_parser("schema", help="print the ModelSpec JSON Schema")
    p_schema.set_defaults(func=_cmd_schema)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
