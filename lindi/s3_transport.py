"""Credentialed S3 byte ranges for the transport-neutral reader."""

from __future__ import annotations

import re
import threading
import time
from urllib.parse import unquote, urlsplit

import boto3
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    IncompleteReadError,
    ReadTimeoutError,
    ResponseStreamingError,
)

from .range_reading import (
    PermanentRangeTransportError,
    RangeTransport,
    RangeTransportCapabilities,
    RangeTransportResult,
    RetryableRangeTransportError,
)


class S3RangeTransport(RangeTransport):
    def __init__(
        self,
        *,
        endpoint_url: str | None = None,
        profile_name: str | None = None,
        region_name: str | None = None,
        max_pool_connections: int = 8,
    ):
        """
        Read signed S3 ranges using standard boto3 credential discovery

        Parameters
        -----------
        endpoint_url: optional s3-compatible service endpoint
        profile_name: optional local aws profile
        region_name: optional signing region
        max_pool_connections: reusable connections per client
        """
        self._session = boto3.Session(profile_name=profile_name, region_name=region_name)
        self._endpoint_url = endpoint_url
        self._max_pool_connections = max_pool_connections
        self._clients = {}
        self._identities = {}
        self._lock = threading.Lock()

    @property
    def capabilities(self) -> RangeTransportCapabilities:
        """
        Describe the S3 transport

        Returns
        --------
        concurrency, caching and source identity capabilities
        """
        return RangeTransportCapabilities(
            name="s3", cacheable=True, high_latency=True,
            supports_concurrent_reads=True, supports_source_identity=True,
        )

    def read_range(
        self, source: str, offset: int, length: int, *, timeout_seconds: float | None,
    ) -> RangeTransportResult:
        """
        Perform one signed, integrity-checked range attempt

        Parameters
        -----------
        source: s3 object url
        offset: first byte
        length: requested byte count
        timeout_seconds: connect and socket read timeout

        Returns
        --------
        exact bytes and transport measurements
        """
        parsed = urlsplit(source)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
            raise PermanentRangeTransportError("Expected an s3 object URL")
        started = time.perf_counter()
        data = b""
        try:
            with self._lock:
                client = self._clients.get(timeout_seconds)
                if client is None:
                    client = self._session.client(
                        "s3", endpoint_url=self._endpoint_url,
                        config=Config(
                            connect_timeout=timeout_seconds, read_timeout=timeout_seconds,
                            max_pool_connections=self._max_pool_connections,
                            # The range executor owns retry policy and metrics.
                            retries={"total_max_attempts": 1},
                            s3={"addressing_style": "path"},
                        ),
                    )
                    self._clients[timeout_seconds] = client
                identity = self._identities.get(source)
            request = {
                "Bucket": parsed.netloc, "Key": unquote(parsed.path[1:]),
                "Range": f"bytes={offset}-{offset + length - 1}",
            }
            if identity is not None:
                request["IfMatch"] = identity
            response = client.get_object(**request)
            body = response["Body"]
            try:
                content_range = response.get("ContentRange", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if (
                    response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 206
                    or match is None
                    or int(match[1]) != offset
                    or int(match[2]) != offset + length - 1
                    or int(match[3]) <= int(match[2])
                    or response.get("ContentLength") != length
                ):
                    raise RetryableRangeTransportError("S3 returned an inconsistent byte range")
                data = body.read(length + 1)
                if len(data) != length:
                    raise RetryableRangeTransportError(
                        "S3 returned an incomplete byte range", fetched_bytes=len(data),
                    )
            finally:
                body.close()
            response_identity = response.get("ETag")
            if not response_identity:
                raise PermanentRangeTransportError("S3 response did not include an ETag")
            with self._lock:
                recorded = self._identities.setdefault(source, response_identity)
            if response_identity != recorded:
                raise PermanentRangeTransportError("S3 object changed during range reads")
        except ClientError as error:
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            failure = (
                RetryableRangeTransportError
                if status in (408, 429, 500, 502, 503, 504)
                else PermanentRangeTransportError
            )
            raise failure(f"S3 range request failed with status {status}") from None
        except (
            EndpointConnectionError, ConnectionClosedError, ConnectTimeoutError,
            ReadTimeoutError, IncompleteReadError, ResponseStreamingError,
        ):
            raise RetryableRangeTransportError(
                "S3 range connection or response failed", fetched_bytes=len(data),
            ) from None
        except BotoCoreError:
            raise PermanentRangeTransportError("S3 credentials or client request failed") from None
        return RangeTransportResult(
            data=data, fetched_bytes=len(data), source_identity=response_identity,
            elapsed_seconds=time.perf_counter() - started,
        )

    def close(self) -> None:
        """
        Close reusable S3 clients
        """
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            client.close()
