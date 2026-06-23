"""Batch extraction — run the pipeline over many targets, fault-tolerant.

Each target (a HF repo id or a local dir) is extracted independently; a failure
is recorded, never fatal. Extraction is IO-bound (metadata downloads), so we run
several targets concurrently. A per-target wall-clock timeout means a single
slow / huge / hung repo (e.g. a 685B model with 160+ shards) is recorded as a
failure and skipped instead of stalling the whole batch. See docs/analytics.md.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from modelspec.pipeline import extract
from modelspec.schema import ModelSpec

# Default per-target wall-clock budget (seconds). Generous enough for a normal
# sharded model, short enough that one pathological repo can't hang a batch.
DEFAULT_TARGET_TIMEOUT = 120.0


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
    target_timeout: float | None = DEFAULT_TARGET_TIMEOUT,
) -> BatchResult:
    """Extract every target concurrently and collect the outcomes.

    Results preserve the input order. ``on_progress(done, total)`` is invoked
    after each completion. Each target runs in its own daemon thread with a
    ``target_timeout`` (seconds) wall-clock budget; a target that exceeds it is
    recorded as a ``TimeoutError`` failure and the batch moves on (the daemon
    thread, if still doing network IO, won't block process exit). Pass
    ``target_timeout=None`` (or <= 0) to disable the timeout.
    """
    targets = list(targets)
    if limit is not None:
        targets = targets[:limit]
    total = len(targets)
    workers = max(1, max_workers)
    budget = target_timeout if (target_timeout and target_timeout > 0) else None

    results: dict[int, BatchItem] = {}
    lock = threading.Lock()
    done = 0

    def _worker(idx: int, target: str) -> None:
        item = _extract_one(target, revision=revision, offline=offline)
        with lock:
            results.setdefault(idx, item)  # ignore if already marked timed-out

    # Process in waves of `workers`; join each wave against a shared deadline so a
    # stuck target is abandoned (daemon) and recorded as a timeout failure.
    for start in range(0, total, workers):
        wave = list(enumerate(targets[start : start + workers], start=start))
        threads = [
            (idx, target, threading.Thread(target=_worker, args=(idx, target), daemon=True))
            for idx, target in wave
        ]
        for _, _, th in threads:
            th.start()
        deadline = time.monotonic() + budget if budget else None
        for idx, target, th in threads:
            timeout = max(0.0, deadline - time.monotonic()) if deadline else None
            th.join(timeout)
            with lock:
                if idx not in results:
                    results[idx] = BatchItem(
                        target=target,
                        error=f"TimeoutError: exceeded {target_timeout}s (slow/hung target)",
                    )
                done += 1
            if on_progress is not None:
                on_progress(done, total)

    return BatchResult(items=[results[i] for i in range(total)])
