"""Shared connection configuration for all database command entries."""

import os
from pathlib import Path

import yaml
from sqlalchemy.engine import URL


DEFAULT_CONFIG = Path(__file__).with_name("config") / "database.yaml"


def get_database_url(config_path: Path | None = None) -> str:
    """Environment overrides the local file; build URLs without exposing secrets."""
    if url := os.environ.get("DATABASE_URL"):
        return url
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG
    if not path.is_file():
        raise ValueError("请配置 database/config/database.yaml，或设置 DATABASE_URL")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        raise ValueError("数据库配置 YAML 格式错误，请检查 database.yaml") from None
    if not isinstance(data, dict) or not isinstance(data.get("postgresql"), dict):
        raise ValueError("database.yaml 需要 postgresql 配置段")
    settings = data["postgresql"]
    for name in ("host", "database", "username", "password"):
        if not isinstance(settings.get(name), str) or not settings[name]:
            raise ValueError(f"请在 database.yaml 中填写 postgresql.{name}")
    port = settings.get("port", 5432)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("postgresql.port 必须是 1 到 65535 的整数")
    return URL.create("postgresql+psycopg", username=settings["username"],
                      password=settings["password"], host=settings["host"], port=port,
                      database=settings["database"]).render_as_string(hide_password=False)


def database_url_or_error(parser) -> str:
    """Present missing/invalid configuration as a concise command-line error."""
    try:
        return get_database_url()
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
