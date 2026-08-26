#!/usr/bin/env python3
"""Unit tests for the Bad Apple dashboard web server."""

import time
import unittest
import urllib.request

import badapple_dashboard


class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.web = badapple_dashboard.DashboardWebServer("127.0.0.1", 0)
        self.web.start(None)
        # Wait briefly for the server to start accepting connections.
        for _ in range(50):
            if self.web._server is not None:
                break
            time.sleep(0.01)
        self.port = self.web._server.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self) -> None:
        if self.web._server is not None:
            self.web._server.shutdown()
            self.web._server.server_close()

    def _get(self, path: str, timeout: int = 5) -> tuple[int, str]:
        url = f"{self.base}{path}"
        req = urllib.request.Request(url, method="GET", headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        return resp.status, body

    def test_index_serves_html(self) -> None:
        """GET / returns the dashboard HTML."""
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("Bad Apple", body)

    def test_static_css_is_served(self) -> None:
        """GET /static/styles.css returns CSS."""
        status, body = self._get("/static/styles.css")
        self.assertEqual(status, 200)
        self.assertIn("body", body)

    def test_status_endpoint_without_daemon(self) -> None:
        """GET /api/status returns a JSON payload even without a daemon."""
        status, body = self._get("/api/status")
        self.assertEqual(status, 200)
        self.assertIn('"runtime"', body)


if __name__ == "__main__":
    unittest.main()
