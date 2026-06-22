"""Batch extraction — run the pipeline over many targets, fault-tolerant.

Each target (a HF repo id or a local dir) is extracted independently; a failure
is recorded, never fatal. Extraction is IO-bound (metadata downloads), so a
thread pool gives good throughput. See docs/analytics.md.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from modelspec.pipeline import extract
from modelspec.schema import ModelSpec


@dataclass
class BatchItem:
    """The outcome of extracting one target."""

    target: str
    spec: ModelSpec | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.spec is not None


@dataclass
class BatchResult:
    items: list[BatchItem] = field(default_factory=list)

    @property
    def specs(self) -> list[ModelSpec]:
        return [it.spec for it in self.items if it.spec is not None]

    @property
    def failures(self) -> list[BatchItem]:
        return [it for it in self.items if it.error is not None]

    @property
    def succeeded(self) -> int:
        return len(self.specs)

    @property
    def failed(self) -> int:
        return len(self.failures)

    @property
    def total(self) -> int:
        return len(self.items)


def read_targets(source: str | Path) -> list[str]:
    """Read targets from a file (one per line), or stdin when source is ``-``.

    Blank lines and ``#`` comments are skipped; inline comments are stripped.
    """
    if str(source) == "-":
        text = sys.stdin.read()
    else:
        text = Path(source).read_text(encoding="utf-8")
    targets: list[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            targets.append(line)
    return targets


def _extract_one(target: str, *, revision: str | None, offline: bool) -> BatchItem:
    try:
        spec = extract(target, revision=revision, offline=offline)
        return BatchItem(target=target, spec=spec)
    except Exception as e:  # noqa: BLE001 - batch must never abort on one target
        return BatchItem(target=target, error=f"{type(e).__name__}: {e}")


def run_batch(
    targets: Iterable[str],
    *,
    offline: bool = False,
    revision: str | None = None,
    max_workers: int = 8,
    limit: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """Extract every target concurrently and collect the outcomes.

    Results preserve the input order. ``on_progress(done, total)`` is invoked
    after each completion (useful for a CLI progress line).
    """
    targets = list(targets)
    if limit is not None:
        targets = targets[:limit]
    total = len(targets)

    results: dict[int, BatchItem] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {
            pool.submit(_extract_one, t, revision=revision, offline=offline): i
            for i, t in enumerate(targets)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
            done += 1
            if on_progress is not None:
                on_progress(done, total)

    return BatchResult(items=[results[i] for i in range(total)])
