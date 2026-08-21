import re

import pytest
import requests

from lindi.range_reading import (
    HttpRangeTransport,
    PermanentRangeTransportError,
    RetryableRangeTransportError,
)


class _FakeResponse:
    def __init__(self, *, status, data, offset, etag):
        self.status_code = status
        self._data = data
        self.headers = {}
        if status == 206:
            self.headers.update(
                {
                    "Content-Range": (
                        f"bytes {offset}-{offset + len(data) - 1}/8"
                    ),
                    "Content-Length": str(len(data)),
                }
            )
        if etag is not None:
            self.headers["ETag"] = etag

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        del chunk_size
        yield self._data


def _response_sequence(monkeypatch, specs):
    calls = []
    responses = iter(specs)

    def get(_session, _url, *, headers, timeout, stream):
        del timeout
        assert stream is True
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"])
        assert match is not None
        offset = int(match.group(1))
        length = int(match.group(2)) - offset + 1
        status, etag = next(responses)
        calls.append(dict(headers))
        return _FakeResponse(
            status=status,
            data=bytes(range(offset, offset + length)),
            offset=offset,
            etag=etag,
        )

    monkeypatch.setattr(requests.Session, "get", get)
    return calls


def test_http_etag_is_reused_and_identity_changes_are_rejected(monkeypatch):
    calls = _response_sequence(
        monkeypatch,
        [(206, '"version-1"'), (206, None), (206, '"version-2"')],
    )
    transport = HttpRangeTransport()
    first = transport.read_range(
        "https://example.test/data", 0, 4, timeout_seconds=1
    )
    second = transport.read_range(
        "https://example.test/data", 4, 4, timeout_seconds=1
    )
    assert first.source_identity == '"version-1"'
    assert second.source_identity == '"version-1"'
    assert "If-Match" not in calls[0]
    assert calls[1]["If-Match"] == '"version-1"'
    with pytest.raises(PermanentRangeTransportError, match="Source changed"):
        transport.read_range(
            "https://example.test/data", 0, 4, timeout_seconds=1
        )
    transport.close()


def test_http_precondition_failure_is_permanent(monkeypatch):
    calls = _response_sequence(
        monkeypatch,
        [(206, '"version-1"'), (412, '"version-2"')],
    )
    transport = HttpRangeTransport()
    transport.read_range("https://example.test/data", 0, 4, timeout_seconds=1)
    with pytest.raises(PermanentRangeTransportError, match="Source changed"):
        transport.read_range(
            "https://example.test/data", 4, 4, timeout_seconds=1
        )
    assert calls[1]["If-Match"] == '"version-1"'
    transport.close()


@pytest.mark.parametrize(
    "error_type",
    [
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ContentDecodingError,
    ],
)
def test_http_connection_failures_are_retryable(monkeypatch, error_type):
    def fail(*args, **kwargs):
        del args, kwargs
        raise error_type("imposed connection failure")

    monkeypatch.setattr(requests.Session, "get", fail)
    transport = HttpRangeTransport()
    with pytest.raises(RetryableRangeTransportError):
        transport.read_range(
            "https://example.test/data", 0, 4, timeout_seconds=1
        )
    transport.close()


def test_unclassified_requests_failure_is_permanent(monkeypatch):
    def fail(*args, **kwargs):
        del args, kwargs
        raise requests.RequestException("imposed request failure")

    monkeypatch.setattr(requests.Session, "get", fail)
    transport = HttpRangeTransport()
    with pytest.raises(PermanentRangeTransportError):
        transport.read_range(
            "https://example.test/data", 0, 4, timeout_seconds=1
        )
    transport.close()
