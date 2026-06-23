# Command-Line Interface

> **Status**: `extract` (with `--format` / `-o` / `--offline` / `--revision` / `--show-provenance` / `--strict`), `schema`, `batch`, and `coverage` are all implemented (`modelspec/cli.py`). `extract` already wires in the six source types + quantization/merge + cross-validation. `--db` was cancelled; `--no-license-llm` (the third license tier has no model wired up) is currently a no-op.

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
