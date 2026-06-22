"""hf_fetcher — guard against a server that ignores Range and returns 200.

No network: a fake session yields far more bytes than requested, simulating a
CDN that sends the whole (multi-GB) file. _read_prefix must stop early.
"""

from __future__ import annotations

from modelspec.io.hf_fetcher import _read_prefix


class _FakeResponse:
    def __init__(self, total_bytes: int):
        self._total = total_bytes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size: int):
        # Pretend the server ignored Range and is streaming a huge file.
        served = 0
        while served < self._total:
            n = min(chunk_size, self._total - served)
            served += n
            yield b"\x00" * n


class _FakeSession:
    def __init__(self, total_bytes: int):
        self._total = total_bytes
        self.requested_ranges: list[str] = []

    def get(self, url, headers=None, stream=False, timeout=None):
        if headers:
            self.requested_ranges.append(headers.get("Range", ""))
        return _FakeResponse(self._total)


def test_read_prefix_stops_early_when_range_ignored():
    # Server would stream 1 GB; we only want 8 bytes.
    session = _FakeSession(total_bytes=1 << 30)
    data = _read_prefix(session, "http://x", 8, timeout=10)
    assert len(data) == 8
    assert session.requested_ranges == ["bytes=0-7"]


def test_read_prefix_returns_available_when_file_smaller():
    session = _FakeSession(total_bytes=100)
    data = _read_prefix(session, "http://x", 4096, timeout=10)
    assert len(data) == 100  # file shorter than the requested prefix
