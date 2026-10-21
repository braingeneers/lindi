import io
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

from lindi import LindiRangeReadOptions, PermanentRangeTransportError, RetryableRangeTransportError
from lindi.range_reading import ByteRange, PlannedByteRangeRequest, fetch_byte_range_requests
from lindi.s3_transport import S3RangeTransport


def _response(data=b"abcd", **changes):
    response = {
        "Body": io.BytesIO(data), "ETag": '"v1"',
        "ContentRange": "bytes 2-5/10", "ContentLength": 4,
        "ResponseMetadata": {"HTTPStatusCode": 206},
    }
    response.update(changes)
    return response


@pytest.fixture
def transport(monkeypatch):
    client = Mock()
    session = Mock()
    session.client.return_value = client
    monkeypatch.setattr("lindi.s3_transport.boto3.Session", Mock(return_value=session))
    reader = S3RangeTransport(endpoint_url="https://s3.example", profile_name="pilot")
    yield reader, client, session
    reader.close()


def test_signed_range_identity_and_cleanup(transport):
    reader, client, session = transport
    first, second = _response(), _response()
    client.get_object.side_effect = [first, second]
    for _ in range(2):
        result = reader.read_range("s3://bucket/a%20b.nwb", 2, 4, timeout_seconds=12)
        assert result.data == b"abcd"
        assert result.source_identity == '"v1"'
    assert first["Body"].closed and second["Body"].closed
    assert client.get_object.call_args.kwargs == {
        "Bucket": "bucket", "Key": "a b.nwb", "Range": "bytes=2-5", "IfMatch": '"v1"',
    }
    config = session.client.call_args.kwargs["config"]
    assert config.retries["total_max_attempts"] == 1
    assert config.connect_timeout == config.read_timeout == 12
    reader.close()
    client.close.assert_called_once()


@pytest.mark.parametrize("changes", [
    {"ContentRange": "bytes 0-3/10"}, {"ContentRange": "bytes 2-5/5"},
    {"ContentLength": 10}, {"ResponseMetadata": {"HTTPStatusCode": 200}},
    {"Body": None},
])
def test_response_integrity(transport, changes):
    reader, client, _ = transport
    if changes.get("Body", True) is None:
        changes = {"Body": io.BytesIO(b"ab")}
    response = _response(**changes)
    client.get_object.return_value = response
    with pytest.raises(RetryableRangeTransportError):
        reader.read_range("s3://bucket/key", 2, 4, timeout_seconds=1)
    assert response["Body"].closed


def test_changed_etag_is_permanent(transport):
    reader, client, _ = transport
    client.get_object.side_effect = [_response(), _response(ETag='"v2"')]
    reader.read_range("s3://bucket/key", 2, 4, timeout_seconds=1)
    with pytest.raises(PermanentRangeTransportError, match="changed"):
        reader.read_range("s3://bucket/key", 2, 4, timeout_seconds=1)


@pytest.mark.parametrize("status", [403, 404, 412, 416, 503])
def test_s3_failure_classification(transport, status):
    reader, client, _ = transport
    client.get_object.side_effect = ClientError({
        "Error": {"Code": "Denied", "Message": "credential-bearing service detail"},
        "ResponseMetadata": {"HTTPStatusCode": status},
    }, "GetObject")
    failure = RetryableRangeTransportError if status == 503 else PermanentRangeTransportError
    with pytest.raises(failure, match=f"status {status}") as caught:
        reader.read_range("s3://bucket/key", 2, 4, timeout_seconds=1)
    assert "credential-bearing" not in str(caught.value)


def test_executor_counts_s3_retries(transport):
    reader, client, _ = transport
    client.get_object.side_effect = [ClientError({
        "Error": {"Code": "SlowDown"}, "ResponseMetadata": {"HTTPStatusCode": 503},
    }, "GetObject"), _response()]
    ref = ByteRange("data/0", "s3://bucket/key", 2, 4)
    request = PlannedByteRangeRequest(ref.source, 2, 4, (ref,), 4, 0)
    results = fetch_byte_range_requests(
        [request], LindiRangeReadOptions(max_retries=1, retry_backoff_seconds=0),
        range_transports={"s3": lambda: reader},
    )
    assert results[0].attempt_count == 2
    assert client.get_object.call_count == 2
