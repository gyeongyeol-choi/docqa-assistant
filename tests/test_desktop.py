"""Unit tests for desktop app helpers and resource/path resolution."""

from pathlib import Path
import socket
import threading
import unittest

from docqa.config import Settings, resolve_resource_path
from docqa.desktop import DesktopBridge, find_free_port, wait_for_server


class _FakeWebview:
    FOLDER_DIALOG = 1

    def __init__(self, selected: list[str] | None) -> None:
        self.selected = selected


class _FakeWindow:
    def __init__(self, selected: list[str] | None) -> None:
        self.selected = selected
        self.last_dialog_type = None

    def create_file_dialog(self, dialog_type: int):
        self.last_dialog_type = dialog_type
        return self.selected


class TestDesktopHelpers(unittest.TestCase):
    def test_resolve_resource_path_finds_static_index(self) -> None:
        path = resolve_resource_path("docqa/static/index.html")
        self.assertTrue(path.exists())

    def test_for_desktop_returns_settings(self) -> None:
        settings = Settings.for_desktop()
        self.assertIsInstance(settings.index_path, Path)
        self.assertIsInstance(settings.docs_dir, Path)

    def test_find_free_port_and_wait_for_server(self) -> None:
        port = find_free_port()
        self.assertGreater(port, 0)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.bind(("127.0.0.1", port))
            srv.listen(1)

            def accept_once() -> None:
                conn, _ = srv.accept()
                conn.close()

            t = threading.Thread(target=accept_once, daemon=True)
            t.start()
            wait_for_server("127.0.0.1", port, timeout=2.0)
            t.join(timeout=1.0)

    def test_desktop_bridge_pick_folder(self) -> None:
        fake_wv = _FakeWebview(["C:/docs"])
        bridge = DesktopBridge(fake_wv)
        self.assertIsNone(bridge.pick_folder())

        window = _FakeWindow(["C:/docs"])
        bridge.attach(window)
        self.assertEqual(bridge.pick_folder(), "C:/docs")
        self.assertEqual(window.last_dialog_type, _FakeWebview.FOLDER_DIALOG)

        bridge.attach(_FakeWindow([]))
        self.assertIsNone(bridge.pick_folder())


if __name__ == "__main__":
    unittest.main()
