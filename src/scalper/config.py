"""Loads secrets from .env and non-secret app config from config/settings.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class OandaCredentials:
    api_token: str
    account_id: str
    environment: str  # "practice" or "live"

    @property
    def rest_host(self) -> str:
        if self.environment == "practice":
            return "https://api-fxpractice.oanda.com"
        return "https://api-fxtrade.oanda.com"

    @property
    def stream_host(self) -> str:
        if self.environment == "practice":
            return "https://stream-fxpractice.oanda.com"
        return "https://stream-fxtrade.oanda.com"


def load_credentials(env_file: Path | None = None) -> OandaCredentials:
    load_dotenv(env_file or PROJECT_ROOT / ".env")

    token = os.environ.get("OANDA_API_TOKEN", "").strip()
    account_id = os.environ.get("OANDA_ACCOUNT_ID", "").strip()
    environment = os.environ.get("OANDA_ENV", "practice").strip() or "practice"

    if not token or not account_id:
        raise RuntimeError(
            "Missing Oanda credentials. Copy .env.example to .env and fill in "
            "OANDA_API_TOKEN and OANDA_ACCOUNT_ID."
        )
    if environment != "practice":
        raise RuntimeError(
            "This app is built for the Oanda practice (demo) environment only. "
            "Set OANDA_ENV=practice in .env."
        )

    return OandaCredentials(api_token=token, account_id=account_id, environment=environment)


def load_settings(settings_file: Path | None = None) -> dict:
    path = settings_file or PROJECT_ROOT / "config" / "settings.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
