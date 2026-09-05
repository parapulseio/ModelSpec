# Command-Line Interface

> **Status**: `extract` (with `--format` / `-o` / `--offline` / `--revision` / `--show-provenance` / `--strict` / `--download-only` / `--analysis-only` / `--output-dir`), `schema`, `batch`, and `coverage` are all implemented (`modelspec/cli.py`). `extract` already wires in the six source types + quantization/merge + cross-validation. `--db` was cancelled; `--no-license-llm` (the third license tier has no model wired up) is currently a no-op.

## Design goal

One command auto-downloads metadata from the HF Hub and outputs a normalized `ModelSpec`, **without downloading weights**.

## Main command

```bash
modelspec extract <repo_id> [options]
```

Examples:

```bash
modelspec extract meta-llama/Llama-3.1-8B-Instruct
modelspec extract TheBloke/Mistral-7B-v0.1-GGUF --format yaml
modelspec extract ./local/model/dir --offline
```

### Options

| Option | Description |
| --- | --- |
| `--format json\|yaml` | output format, default json |
| `-o, --output PATH` | write to a file, default stdout |
| `--offline` | local paths only, no HF access |
| `--revision REV` | specify a commit / branch / tag |
| `--no-license-llm` | disable the third LLM tier of license identification |
| `--show-provenance` | include per-field provenance and conflicts in the output |
| `--strict` | non-zero exit on a validation issue (for CI) |
| `--download-only` | only fetch metadata to `--output-dir` and write a manifest; don't analyze |
| `--analysis-only` | only analyze `repo_id` as an already-downloaded local directory; no network |
| `--output-dir DIR` | destination directory for `--download-only` (default: `./<repo_id>`) |

### Splitting download from analysis

`--download-only` and `--analysis-only` split the normal one-shot `extract` into two steps — useful for auditing exactly what left the network, or for running analysis repeatedly (schema changes, debugging) without re-fetching:

```bash
# fetch metadata only; writes files under ./meta-llama/Llama-3.1-8B/
# plus a MODELSPEC_MANIFEST.md describing what was pulled and how
modelspec extract meta-llama/Llama-3.1-8B --download-only

# analyze that directory later, fully offline
modelspec extract meta-llama/Llama-3.1-8B --analysis-only
```

`--output-dir` defaults to `./<repo_id>` (mirroring the `org/name` path), so the same `repo_id` string works for both commands without repeating a path. `MODELSPEC_MANIFEST.md` records the original `repo_id` / `revision`, which `--analysis-only` (and offline `extract` on any local directory containing this file) reads back so `identity.repo_id` reflects the real Hub repo instead of the local path. The manifest also lists every file that landed on disk, how it was fetched (full download vs. header-only Range request), which weight files were skipped, and copy-pasteable commands for the two follow-up moves: analyze as-is, or re-run `--download-only` to refresh a stale copy.

## Auxiliary commands

```bash
modelspec schema             # export the JSON Schema (ModelSpec.model_json_schema())
modelspec batch repos.txt    # batch extraction + an unknown_fields frequency report (M4, implemented)
modelspec coverage repos.txt # the coverage sanity-check dashboard (M4, implemented)
modelspec explain <field>    # explain what a schema field means (M5)
modelspec completion <shell> # print a bash/zsh/fish tab-completion script (M5)
```

The full options and field-promotion workflow for `batch` / `coverage` are in [analytics.md](analytics.md).

### `explain` — field documentation (M5)

The schema is self-documenting: every field carries a `description=`, so `explain`
introspects the live `ModelSpec` and prints the type, allowed values and description.
Matching is fuzzy — an exact dotted path wins, else a bare leaf name, else any substring.

```bash
modelspec explain context.effective     # exact dotted path
modelspec explain tied_embeddings        # bare leaf name
modelspec explain quant                  # substring -> all matching fields
modelspec explain                        # no arg -> list every field
modelspec explain attention.type --format json
```

The same catalog is available to library consumers via
`modelspec.explain.field_catalog()` / `explain_field()`.

### `completion` — shell tab-completion (M5)

```bash
source <(modelspec completion bash)                # bash (in ~/.bashrc)
source <(modelspec completion zsh)                 # zsh  (in ~/.zshrc)
modelspec completion fish > ~/.config/fish/completions/modelspec.fish
```

The scripts are static (no extra dependency) and complete subcommands plus each
subcommand's options.

## Exit codes

`extract`:

| Code | Meaning |
| --- | --- |
| 0 | success |
| 1 | extraction / validation failed (or `--strict` hit a warning) |
| 2 | repo not found or network error |

`batch` / `coverage`:

| Code | Meaning |
| --- | --- |
| 0 | completed (including partial failures — normal at corpus scale, failures are recorded as data) |
| 1 | everything failed, zero successes |
| 2 | cannot read the targets file |

## Output example (excerpt)

```json
{
  "spec_version": "1.0",
  "identity": { "repo_id": "meta-llama/Llama-3.1-8B-Instruct", "source_format": "hf" },
  "architecture": { "family": "llama", "num_layers": 32, "tags": ["decoder-only", "gqa", "rope-llama3"] },
  "attention": { "type": "gqa", "num_heads": 32, "num_kv_heads": 8 },
  "parameters": { "total": 8030261248, "dtype_native": "BF16" },
  "context": { "declared": 131072, "rope_scaling": { "type": "llama3", "factor": 8.0 } },
  "license": { "spdx_id": "llama3.1", "commercial_use": true, "confidence_tier": "fingerprint" },
  "provenance": { "conflicts": [], "warnings": [], "unknown_fields": ["pretraining_tp"] }
}
```
