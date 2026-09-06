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
from modelspec.schema import ModelSpec, export_json_schema


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


def _finish_extract(spec: ModelSpec, args: argparse.Namespace) -> int:
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


def _cmd_extract_download_only(args: argparse.Namespace) -> int:
    if args.offline:
        print("error: --download-only needs network access; drop --offline", file=sys.stderr)
        return 2
    if Path(args.repo_id).is_dir():
        print(
            f"error: {args.repo_id!r} is a local directory; --download-only fetches "
            "from the HF Hub — pass a repo id instead",
            file=sys.stderr,
        )
        return 2

    from modelspec.io.hf_fetcher import MANIFEST_FILENAME, download_metadata, render_manifest

    dest_dir = Path(args.output_dir) if args.output_dir else Path(args.repo_id)
    try:
        result = download_metadata(args.repo_id, revision=args.revision, dest_dir=dest_dir)
    except Exception as e:  # network / HTTP errors
        print(f"error: download failed: {e}", file=sys.stderr)
        return 1

    manifest = render_manifest(
        repo_id=args.repo_id,
        revision=args.revision,
        dest_dir=dest_dir,
        commit_sha=result.commit_sha,
        entries=result.files,
    )
    (dest_dir / MANIFEST_FILENAME).write_text(manifest, encoding="utf-8")

    n_downloaded = sum(1 for e in result.files if e.kind != "skipped")
    print(f"downloaded {n_downloaded} file(s) to {dest_dir}/", file=sys.stderr)
    print(f"manifest: {dest_dir / MANIFEST_FILENAME}", file=sys.stderr)
    return 0


def _cmd_extract_analysis_only(args: argparse.Namespace) -> int:
    path = Path(args.repo_id)
    if not path.is_dir():
        print(
            f"error: {args.repo_id!r} is not a local directory; --analysis-only "
            "only reads metadata already on disk — run "
            f"'modelspec extract {args.repo_id} --download-only' first",
            file=sys.stderr,
        )
        return 2

    try:
        spec = extract(str(path), offline=True)
    except Exception as e:  # parse / validation
        print(f"error: extraction failed: {e}", file=sys.stderr)
        return 1
    return _finish_extract(spec, args)


def _cmd_extract(args: argparse.Namespace) -> int:
    if args.download_only and args.analysis_only:
        print("error: --download-only and --analysis-only are mutually exclusive", file=sys.stderr)
        return 2
    if args.download_only:
        return _cmd_extract_download_only(args)
    if args.analysis_only:
        return _cmd_extract_analysis_only(args)

    try:
        spec = extract(args.repo_id, revision=args.revision, offline=args.offline)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # network / parse / validation
        print(f"error: extraction failed: {e}", file=sys.stderr)
        return 1
    return _finish_extract(spec, args)


def _cmd_schema(args: argparse.Namespace) -> int:
    print(json.dumps(export_json_schema(), indent=2, ensure_ascii=False))
    return 0


def _infer_doc_format(path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    return "yaml" if path.suffix.lower() in (".yaml", ".yml") else "json"


def _load_document(path: Path, fmt: str) -> Any:
    text = path.read_text(encoding="utf-8")
    if fmt == "yaml":
        import yaml  # already a hard requirement of --format yaml elsewhere

        return yaml.safe_load(text)
    return json.loads(text)


def _cmd_verify(args: argparse.Namespace) -> int:
    """Validate a previously-extracted ModelSpec document against the live schema.

    Independent of `extract`: reads plain JSON/YAML off disk and checks it with
    a standard JSON Schema validator, so it also catches a document that was
    hand-edited, produced by another tool, or extracted with an older/newer
    version of modelspec than the schema installed here.
    """
    try:
        import jsonschema
    except ImportError:
        print(
            "error: verify requires jsonschema (pip install 'modelspec[verify]')",
            file=sys.stderr,
        )
        return 2

    path = Path(args.file)
    fmt = _infer_doc_format(path, args.format)
    try:
        data = _load_document(path, fmt)
    except ImportError:
        print(
            "error: --format yaml requires PyYAML (pip install 'modelspec[yaml]')",
            file=sys.stderr,
        )
        return 2
    except OSError as e:
        print(f"error: cannot read {path}: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # json.JSONDecodeError / yaml.YAMLError
        print(f"error: cannot parse {path} as {fmt}: {e}", file=sys.stderr)
        return 2

    schema = export_json_schema()
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: [str(p) for p in e.path])

    if not errors:
        print(f"valid: {path} conforms to the ModelSpec schema")
        return 0

    print(f"invalid: {path} does not conform to the ModelSpec schema", file=sys.stderr)
    for e in errors:
        loc = "/".join(str(p) for p in e.path) or "<root>"
        print(f"  {loc}: {e.message}", file=sys.stderr)
    return 1


def _cmd_explain(args: argparse.Namespace) -> int:
    from modelspec.explain import explain_field, field_catalog

    if not args.field:
        docs = field_catalog()  # list every field
    else:
        docs = explain_field(args.field)
        if not docs:
            print(f"error: no field matching {args.field!r}", file=sys.stderr)
            print("       run 'modelspec explain' to list all fields", file=sys.stderr)
            return 2

    if args.format == "json":
        print(json.dumps([vars(d) for d in docs], indent=2, ensure_ascii=False))
        return 0

    for d in docs:
        print(f"{d.path}")
        print(f"  type: {d.type_str}")
        if d.choices:
            print(f"  choices: {', '.join(d.choices)}")
        if d.description:
            print(f"  {d.description}")
        print()
    return 0


_COMPLETIONS = {
    "bash": """\
# modelspec bash completion — add to ~/.bashrc:
#   source <(modelspec completion bash)
_modelspec_complete() {
    local cur prev words cword
    _init_completion 2>/dev/null || { cur="${COMP_WORDS[COMP_CWORD]}"; }
    local cmds="extract schema verify batch coverage explain completion"
    if [ "$COMP_CWORD" -eq 1 ]; then
        COMPREPLY=( $(compgen -W "$cmds --help --version" -- "$cur") )
        return
    fi
    case "${COMP_WORDS[1]}" in
        extract) COMPREPLY=( $(compgen -W "--format -o --output --offline --revision --show-provenance --strict --download-only --analysis-only --output-dir" -- "$cur") );;
        verify) COMPREPLY=( $(compgen -W "--format" -- "$cur") );;
        batch|coverage) COMPREPLY=( $(compgen -W "--offline --revision --workers --limit --target-timeout --delay --format --top --quiet --output-dir" -- "$cur") );;
        explain) COMPREPLY=( $(compgen -W "--format" -- "$cur") );;
        completion) COMPREPLY=( $(compgen -W "bash zsh fish" -- "$cur") );;
    esac
}
complete -F _modelspec_complete modelspec
""",
    "zsh": """\
# modelspec zsh completion — add to ~/.zshrc:
#   source <(modelspec completion zsh)
_modelspec() {
    local -a cmds
    cmds=(extract schema verify batch coverage explain completion)
    if (( CURRENT == 2 )); then
        compadd -- $cmds --help --version
        return
    fi
    case $words[2] in
        extract) compadd -- --format -o --output --offline --revision --show-provenance --strict --download-only --analysis-only --output-dir;;
        verify) compadd -- --format;;
        batch|coverage) compadd -- --offline --revision --workers --limit --target-timeout --delay --format --top --quiet --output-dir;;
        explain) compadd -- --format;;
        completion) compadd -- bash zsh fish;;
    esac
}
compdef _modelspec modelspec
""",
    "fish": """\
# modelspec fish completion — add to ~/.config/fish/completions/modelspec.fish:
#   modelspec completion fish > ~/.config/fish/completions/modelspec.fish
complete -c modelspec -f
complete -c modelspec -n __fish_use_subcommand -a extract -d 'extract a ModelSpec'
complete -c modelspec -n __fish_use_subcommand -a schema -d 'print the JSON Schema'
complete -c modelspec -n __fish_use_subcommand -a verify -d 'validate a ModelSpec file against the schema'
complete -c modelspec -n __fish_use_subcommand -a batch -d 'batch extract + report'
complete -c modelspec -n __fish_use_subcommand -a coverage -d 'coverage dashboard'
complete -c modelspec -n __fish_use_subcommand -a explain -d 'explain a schema field'
complete -c modelspec -n __fish_use_subcommand -a completion -d 'print a shell completion script'
""",
}


def _cmd_completion(args: argparse.Namespace) -> int:
    print(_COMPLETIONS[args.shell], end="")
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
        target_timeout=args.target_timeout,
        delay=args.delay,
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
    parser.add_argument(
        "--target-timeout",
        type=float,
        default=120.0,
        help="per-target seconds budget; slow/hung targets are recorded as failures (0 = no limit)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="seconds to wait between starting each target, to throttle the Hub request rate",
    )
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--top", type=int, default=20, help="rows for unknown_fields tables")
    parser.add_argument("--quiet", action="store_true", help="suppress the progress line")


_EXAMPLES = """\
examples:
  # extract one model from the HF Hub (metadata only, no weights downloaded)
  modelspec extract meta-llama/Llama-3.1-8B-Instruct

  # a GGUF repo, as YAML, written to a file
  modelspec extract TheBloke/Mistral-7B-v0.1-GGUF --format yaml -o spec.yaml

  # a local directory, offline, failing CI on any cross-field warning
  modelspec extract ./models/my-model --offline --strict

  # split download from analysis: fetch metadata + write a manifest, no analysis
  modelspec extract meta-llama/Llama-3.1-8B-Instruct --download-only
  # ... then analyze the downloaded directory, no network
  modelspec extract meta-llama/Llama-3.1-8B-Instruct --analysis-only

  # validate an extracted (or hand-edited, or third-party) ModelSpec file
  modelspec verify spec.yaml --format yaml

  # batch a corpus and see which raw keys are still uncovered
  modelspec batch repos.txt --top 30

  # what does a field mean? (fuzzy: bare leaf names work too)
  modelspec explain effective
  modelspec explain attention.num_kv_heads

  # install tab-completion for your shell
  source <(modelspec completion bash)
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modelspec",
        description="Extract and normalize LLM model specifications from their "
        "metadata sources (config.json / safetensors / GGUF / license / "
        "tokenizer) — without downloading weights.",
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"modelspec {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    p_extract = sub.add_parser(
        "extract",
        help="extract a ModelSpec from a repo id or local dir",
        description="Download a model's metadata and emit a normalized ModelSpec.",
        epilog="examples:\n"
        "  modelspec extract meta-llama/Llama-3.1-8B-Instruct\n"
        "  modelspec extract ./local/dir --offline --show-provenance\n"
        "  modelspec extract meta-llama/Llama-3.1-8B-Instruct --download-only\n"
        "  modelspec extract meta-llama/Llama-3.1-8B-Instruct --analysis-only\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    p_extract.add_argument(
        "--download-only",
        action="store_true",
        help="only download metadata to --output-dir + write a manifest; don't analyze",
    )
    p_extract.add_argument(
        "--analysis-only",
        action="store_true",
        help="only analyze repo_id as an already-downloaded local directory; no network",
    )
    p_extract.add_argument(
        "--output-dir",
        help="destination directory for --download-only (default: ./<repo_id>)",
    )
    p_extract.set_defaults(func=_cmd_extract)

    p_schema = sub.add_parser("schema", help="print the ModelSpec JSON Schema")
    p_schema.set_defaults(func=_cmd_schema)

    p_verify = sub.add_parser(
        "verify",
        help="validate a ModelSpec JSON/YAML file against the schema",
        description="Check a previously-extracted (or hand-edited, or "
        "third-party) ModelSpec document against the live JSON Schema — "
        "catches type errors, bad enum values, and invalid discriminated "
        "unions (e.g. quantization).",
        epilog="examples:\n"
        "  modelspec verify llama.json\n"
        "  modelspec verify spec.yaml --format yaml\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_verify.add_argument("file", help="path to a ModelSpec JSON or YAML file")
    p_verify.add_argument(
        "--format",
        choices=["json", "yaml"],
        help="input format (default: infer from file extension, else json)",
    )
    p_verify.set_defaults(func=_cmd_verify)

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

    p_explain = sub.add_parser(
        "explain",
        help="explain what a ModelSpec field means",
        description="Print the type, allowed values and description of a schema "
        "field. With no FIELD, list every field. Matching is fuzzy: an exact "
        "dotted path wins, else a bare leaf name, else any substring.",
        epilog="examples:\n"
        "  modelspec explain effective_context\n"
        "  modelspec explain quant\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_explain.add_argument("field", nargs="?", help="dotted path or name fragment (omit to list all)")
    p_explain.add_argument("--format", choices=["text", "json"], default="text")
    p_explain.set_defaults(func=_cmd_explain)

    p_completion = sub.add_parser(
        "completion",
        help="print a shell tab-completion script",
        description="Emit a completion script for the given shell to stdout.",
        epilog="examples:\n"
        "  source <(modelspec completion bash)\n"
        "  modelspec completion fish > ~/.config/fish/completions/modelspec.fish\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_completion.add_argument("shell", choices=["bash", "zsh", "fish"])
    p_completion.set_defaults(func=_cmd_completion)

    return parser


def _quiet_hf_advisory() -> None:
    """Silence huggingface_hub's per-request advisory log lines.

    The HF server attaches a ``Warning`` header to anonymous requests, which
    huggingface_hub re-logs ("You are sending unauthenticated requests ...").
    At corpus scale this floods the dashboard; our own findings go through
    provenance, not the hub logger, so raising its level loses nothing useful.
    Authenticate (HF_TOKEN) to make the underlying requests faster regardless.
    """
    try:
        from huggingface_hub.utils import logging as _hf_logging

        _hf_logging.set_verbosity_error()
    except Exception:  # pragma: no cover - older/newer hub layouts
        import logging

        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _quiet_hf_advisory()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
