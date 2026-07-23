"""Unit tests for compresr.__main__ — the compresr-sdk CLI."""

from __future__ import annotations

import pytest

from compresr import __main__ as cli
from compresr import credentials


@pytest.fixture
def tmp_creds(tmp_path, monkeypatch):
    path = tmp_path / "creds"
    monkeypatch.setenv("COMPRESR_CREDENTIALS_FILE", str(path))
    monkeypatch.delenv("COMPRESR_API_KEY", raising=False)
    monkeypatch.delenv("COMPRESR_PROFILE", raising=False)
    yield path


class TestVersion:
    def test_version_uses_compresr_sdk_prefix(self, capsys):
        with pytest.raises(SystemExit):
            cli.main(["--version"])
        captured = capsys.readouterr()
        assert captured.out.startswith("compresr-sdk ")


class TestWhoami:
    def test_whoami_when_not_logged_in(self, tmp_creds, capsys):
        rc = cli.main(["whoami"])
        assert rc == 1
        # "Not logged in" is an error condition → stderr, not stdout.
        captured = capsys.readouterr()
        assert "Not logged in" in captured.err
        assert captured.out == ""

    def test_whoami_when_logged_in(self, tmp_creds, capsys):
        credentials.save("cmp_abcdefgh_test_XYZW", profile="default")
        rc = cli.main(["whoami"])
        assert rc == 0
        captured = capsys.readouterr()
        # whoami masks the middle: shows first 8 + ellipsis + last 4.
        assert "cmp_abcd" in captured.out
        assert "XYZW" in captured.out
        assert captured.err == ""


class TestLogout:
    def test_logout_removes_key(self, tmp_creds, capsys):
        credentials.save("cmp_x_XXXXXXXXXXXXXX", profile="default")
        rc = cli.main(["logout"])
        assert rc == 0
        assert credentials.load("default") is None
        assert "Removed credentials" in capsys.readouterr().out

    def test_logout_idempotent_when_missing(self, tmp_creds, capsys):
        # Absent-is-ok: rerunning logout on an already-clean profile is a
        # legitimate operation (scripts, teardown), so exit 0.
        rc = cli.main(["logout"])
        assert rc == 0


class TestStatus:
    def test_status_shows_missing(self, tmp_creds, capsys):
        rc = cli.main(["status"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "credentials file" in out
        assert "exists:           False" in out

    def test_status_enumerates_profiles(self, tmp_creds, capsys):
        credentials.save("cmp_a_XXXXXXXXXXXXXX", profile="default")
        credentials.save("cmp_b_XXXXXXXXXXXXXX", profile="work")
        rc = cli.main(["status"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "default" in out
        assert "work" in out


class TestKeyboardInterrupt:
    def test_ctrl_c_returns_130(self, tmp_creds, monkeypatch, capsys):
        def boom(*_args, **_kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "_login", boom)
        rc = cli.main(["login"])
        assert rc == 130
        assert "Aborted" in capsys.readouterr().err


class TestLoginErrorRouting:
    def test_login_error_goes_to_stderr(self, tmp_creds, monkeypatch, capsys):
        def boom(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(cli, "_login", boom)
        rc = cli.main(["login"])
        assert rc == 1
        out = capsys.readouterr()
        assert "boom" in out.err
        assert out.out == ""
