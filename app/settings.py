import os
from dataclasses import dataclass
from pathlib import Path

from churnguard import config


@dataclass(frozen=True)
class Settings:
    api_key: str
    log_dir: Path


def load_settings():
    return Settings(
        api_key=os.getenv("CHURNGUARD_API_KEY", "demo-key"),
        log_dir=Path(os.getenv("CHURNGUARD_LOG_DIR", str(config.ROOT / "logs"))),
    )
