"""Command-line interface: ``modelspec extract <repo_id|path>``.

See docs/cli.md for the full option reference.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
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


def _progress_printer(done: int, total: int) -> None:
    print(f"\r  extracting {done}/{total} ...", end="", file=sys.stderr, flush=True)
    if done == total:
        print("", file=sys.stderr)


def _run_batch_from_args(args: argparse.Namespace):
    """Shared batch driver for the batch / coverage subcommands."""
    from modelspec.analytics import read_targets, run_batch

    try:
        targets = read_targets(args.targets)
    except OSError as e:
        print(f"error: cannot read targets: {e}", file=sys.stderr)
        return None
    progress = None if args.quiet else _progress_printer
    return run_batch(
        targets,
        offline=args.offline,
        revision=args.revision,
        max_workers=args.workers,
        limit=args.limit,
        on_progress=progress,
    )


def _cmd_batch(args: argparse.Namespace) -> int:
    from modelspec.analytics import build_coverage_report

    result = _run_batch_from_args(args)
    if result is None:
        return 2

    # Optionally persist each spec as JSON (filename = sanitized target).
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for item in result.items:
            if item.spec is None:
                continue
            name = re.sub(r"[^\w.-]", "__", item.target).strip("_") or "model"
            (out_dir / f"{name}.json").write_text(
                item.spec.model_dump_json(indent=2), encoding="utf-8"
            )

    report = build_coverage_report(result, top_n=args.top)
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        # batch focuses on extraction + the unknown_fields signal.
        print(report.render(full=False, top_n=args.top))
    return _exit_code(result)


def _cmd_coverage(args: argparse.Namespace) -> int:
    from modelspec.analytics import build_coverage_report

    result = _run_batch_from_args(args)
    if result is None:
        return 2

    report = build_coverage_report(result, top_n=args.top)
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(report.render(full=True, top_n=args.top))
    return _exit_code(result)


def _exit_code(result) -> int:
    # Partial failures are expected at corpus scale and are reported as data, so
    # they do not fail the command. Only a total wipeout (nothing extracted) is
    # treated as an error.
    if result.total > 0 and result.succeeded == 0:
        return 1
    return 0


def _add_batch_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("targets", help="file of repo ids / paths (one per line; '-' for stdin)")
    parser.add_argument("--offline", action="store_true", help="local paths only, no network")
    parser.add_argument("--revision", help="commit / branch / tag")
    parser.add_argument("--workers", type=int, default=8, help="concurrent extractions")
    parser.add_argument("--limit", type=int, help="only process the first N targets (sampling)")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--top", type=int, default=20, help="rows for unknown_fields tables")
    parser.add_argument("--quiet", action="store_true", help="suppress the progress line")


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

    p_batch = sub.add_parser(
        "batch", help="extract many models; report unknown_fields frequency"
    )
    _add_batch_options(p_batch)
    p_batch.add_argument("--output-dir", help="write each extracted spec as JSON here")
    p_batch.set_defaults(func=_cmd_batch)

    p_coverage = sub.add_parser(
        "coverage", help="extraction coverage sanity check over a corpus"
    )
    _add_batch_options(p_coverage)
    p_coverage.set_defaults(func=_cmd_coverage)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
