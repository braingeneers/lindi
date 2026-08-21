import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture
def serve_file_over_http():
    """Serve immutable files with standards-compliant single-range responses."""
    servers = []

    def serve(path):
        data = path.read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                range_header = self.headers.get("Range")
                if range_header is None:
                    start, end, status = 0, len(data) - 1, 200
                else:
                    match = re.fullmatch(r"bytes=(\d+)-(\d+)", range_header)
                    if match is None:
                        self.send_error(416, "A single explicit byte range is required")
                        return
                    start, end, status = int(match.group(1)), int(match.group(2)), 206
                    if start > end or end >= len(data):
                        self.send_error(416, "Range is outside the fixture")
                        return
                payload = data[start:end + 1]
                self.send_response(status)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(len(payload)))
                if status == 206:
                    self.send_header(
                        "Content-Range", f"bytes {start}-{end}/{len(data)}"
                    )
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, format, *args):
                del format, args

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}/{path.name}"

    yield serve

    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
