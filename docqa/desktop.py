"""Desktop GUI launcher for DocQA Assistant using pywebview."""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

from docqa.config import Settings
from docqa.service import DocQAService


def find_free_port(host: str = "127.0.0.1") -> int:
    """Bind an ephemeral port and return its number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def wait_for_server(host: str, port: int, timeout: float = 5.0) -> None:
    """Poll until the local HTTP server accepts TCP connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"Local server did not start within {timeout:.1f}s on {host}:{port}")


class DesktopBridge:
    """JavaScript-to-Python bridge exposed to the pywebview window."""

    def __init__(self, webview_module: Any) -> None:
        self._webview = webview_module
        self._window: Any = None

    def attach(self, window: Any) -> None:
        """Bind the active pywebview window instance."""
        self._window = window

    def pick_folder(self) -> str | None:
        """Open a native OS folder selection dialog and return the chosen path."""
        if self._window is None:
            return None
        dialog_type = (
            self._webview.FileDialog.FOLDER
            if hasattr(self._webview, "FileDialog")
            else self._webview.FOLDER_DIALOG
        )
        result = self._window.create_file_dialog(dialog_type)
        if result and len(result) > 0:
            return str(result[0])
        return None


def main() -> int:
    """Launch the FastAPI backend in a background thread and open a native window."""
    try:
        import uvicorn
        import webview
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn and pywebview are required for desktop mode. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    from docqa.api import create_app

    host = "127.0.0.1"
    port = find_free_port(host)

    settings = Settings.for_desktop()
    service = DocQAService(settings=settings)
    app = create_app(service=service)

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    try:
        wait_for_server(host, port, timeout=5.0)
        bridge = DesktopBridge(webview)
        window = webview.create_window(
            "DocQA Assistant",
            url=f"http://{host}:{port}",
            width=1020,
            height=760,
            min_size=(760, 560),
            js_api=bridge,
        )
        bridge.attach(window)
        webview.start()
    finally:
        server.should_exit = True
        server_thread.join(timeout=2.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
