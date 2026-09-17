from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from scalper.config import load_credentials, load_settings
from scalper.dashboard.main_window import MainWindow
from scalper.oanda.client import OandaClient


def main() -> int:
    credentials = load_credentials()
    settings = load_settings()
    instruments = settings["instruments"]

    app = QApplication(sys.argv)
    client = OandaClient(credentials)
    window = MainWindow(client, instruments)
    window.show()

    exit_code = app.exec()
    client.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
