import pytest
from sqlalchemy.engine import make_url

from roboego_hand_vis.database.connection_config import get_database_url


def test_environment_takes_precedence(monkeypatch, tmp_path):
    url = "postgresql+psycopg://user:password@localhost/db"
    monkeypatch.setenv("DATABASE_URL", url)
    assert get_database_url(tmp_path / "missing.yaml") == url


def test_file_password_needs_no_manual_url_encoding(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    path = tmp_path / "connection.yaml"
    path.write_text("postgresql:\n  host: localhost\n  port: 5432\n  database: test\n  username: user\n  password: 'a@b:/?#$%!'\n")
    url = make_url(get_database_url(path))
    assert url.drivername == "postgresql+psycopg"
    assert url.password == "a@b:/?#$%!"
    assert url.database == "test"


@pytest.mark.parametrize("contents", ["[]", "postgresql: []", "postgresql: {}", "postgresql:\n  password: ''"])
def test_bad_configuration_is_clear_error(monkeypatch, tmp_path, contents):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    path = tmp_path / "connection.yaml"
    path.write_text(contents)
    with pytest.raises(ValueError):
        get_database_url(path)


def test_missing_file_explains_local_configuration(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="database.yaml"):
        get_database_url(tmp_path / "missing.yaml")
