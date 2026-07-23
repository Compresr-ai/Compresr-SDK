"""Unit tests for compresr.credentials — the ~/.compresr/credentials store."""

from __future__ import annotations

import os

import pytest

from compresr import credentials


@pytest.fixture
def tmp_creds(tmp_path, monkeypatch):
    """Point credentials at a tmpdir file for the duration of one test."""
    path = tmp_path / "creds"
    monkeypatch.setenv("COMPRESR_CREDENTIALS_FILE", str(path))
    monkeypatch.delenv("COMPRESR_API_KEY", raising=False)
    monkeypatch.delenv("COMPRESR_PROFILE", raising=False)
    yield path


class TestSaveAndLoad:
    def test_save_then_load_roundtrip(self, tmp_creds):
        credentials.save("cmp_abc123_test_key_1234", profile="default", base_url="https://x")
        assert credentials.load("default") == "cmp_abc123_test_key_1234"

    def test_load_missing_file_returns_none(self, tmp_creds):
        assert credentials.load("default") is None

    def test_load_missing_profile_returns_none(self, tmp_creds):
        credentials.save("cmp_abc_XXXXXXXXXXXXX", profile="work")
        assert credentials.load("default") is None
        assert credentials.load("work") == "cmp_abc_XXXXXXXXXXXXX"

    def test_save_overwrites_same_profile(self, tmp_creds):
        credentials.save("cmp_old_XXXXXXXXXXXXX", profile="default")
        credentials.save("cmp_new_XXXXXXXXXXXXX", profile="default")
        assert credentials.load("default") == "cmp_new_XXXXXXXXXXXXX"

    def test_multiple_profiles_coexist(self, tmp_creds):
        credentials.save("cmp_a_XXXXXXXXXXXXXX", profile="default")
        credentials.save("cmp_b_XXXXXXXXXXXXXX", profile="work")
        assert credentials.load("default") == "cmp_a_XXXXXXXXXXXXXX"
        assert credentials.load("work") == "cmp_b_XXXXXXXXXXXXXX"

    @pytest.mark.skipif(os.name != "posix", reason="posix perms only")
    def test_save_sets_0600_perms(self, tmp_creds):
        credentials.save("cmp_perm_test_key_1234", profile="default")
        mode = tmp_creds.stat().st_mode & 0o777
        assert mode == 0o600, oct(mode)

    @pytest.mark.skipif(os.name != "posix", reason="posix perms only")
    def test_load_raises_on_world_readable_file(self, tmp_creds):
        credentials.save("cmp_a_XXXXXXXXXXXXXX", profile="default")
        os.chmod(tmp_creds, 0o644)  # world-readable
        # PermissionError propagates so `whoami`/`status` can surface bad perms
        # clearly instead of pretending the user is logged out.
        with pytest.raises(PermissionError):
            credentials.load("default")

    @pytest.mark.skipif(os.name != "posix", reason="posix perms only")
    def test_resolve_api_key_returns_none_on_bad_perms(self, tmp_creds):
        credentials.save("cmp_a_XXXXXXXXXXXXXX", profile="default")
        os.chmod(tmp_creds, 0o644)
        # resolve_api_key stays quiet — CompressionClient construction should
        # fall through to "no key found" rather than crashing on bad perms.
        assert credentials.resolve_api_key(None) is None


class TestClear:
    def test_clear_removes_profile(self, tmp_creds):
        credentials.save("cmp_x_XXXXXXXXXXXXXX", profile="default")
        assert credentials.clear("default") is True
        assert credentials.load("default") is None

    def test_clear_leaves_other_profiles(self, tmp_creds):
        credentials.save("cmp_a_XXXXXXXXXXXXXX", profile="default")
        credentials.save("cmp_b_XXXXXXXXXXXXXX", profile="work")
        credentials.clear("default")
        assert credentials.load("work") == "cmp_b_XXXXXXXXXXXXXX"

    def test_clear_missing_returns_false(self, tmp_creds):
        assert credentials.clear("nothing") is False


class TestTokenShapeValidation:
    def test_rejects_short_key(self, tmp_creds):
        with pytest.raises(ValueError, match="malformed api key"):
            credentials.save("cmp_short")

    def test_rejects_missing_prefix(self, tmp_creds):
        with pytest.raises(ValueError, match="malformed api key"):
            credentials.save("nope_abcdefghijklmnop")

    def test_rejects_overlong_key(self, tmp_creds):
        with pytest.raises(ValueError, match="malformed api key"):
            credentials.save("cmp_" + "a" * 200)

    def test_rejects_control_chars(self, tmp_creds):
        with pytest.raises(ValueError, match="malformed api key"):
            credentials.save("cmp_bad\x00key_with_null_XX")


class TestSymlinkDefense:
    @pytest.mark.skipif(os.name != "posix", reason="posix symlinks only")
    def test_load_refuses_symlink(self, tmp_creds, tmp_path):
        # Point the credentials env at a symlink; load must refuse.
        real_file = tmp_path / "real"
        real_file.write_text("[default]\napi_key = cmp_a_XXXXXXXXXXXXXX\n")
        real_file.chmod(0o600)
        link = tmp_path / "link"
        link.symlink_to(real_file)
        import os as _os

        _os.environ["COMPRESR_CREDENTIALS_FILE"] = str(link)
        with pytest.raises(PermissionError, match="symlink"):
            credentials.load("default")


_LOCK_HOLDER_SCRIPT = (
    "import fcntl, os, sys, time\n"
    "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
    "fcntl.flock(fd, fcntl.LOCK_EX)\n"
    "time.sleep(5)\n"
)


class TestLockDeadline:
    @pytest.mark.skipif(os.name != "posix", reason="posix flock only")
    def test_lock_deadline_raises_on_stale_holder(self, tmp_creds, monkeypatch):
        # Simulate a stuck holder by shortening the deadline and holding the
        # lock from a child process.
        import subprocess
        import time as _time

        lock_path = tmp_creds.with_suffix(tmp_creds.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(["python", "-c", _LOCK_HOLDER_SCRIPT, str(lock_path)])
        try:
            _time.sleep(0.5)  # let the child grab the lock
            monkeypatch.setattr(credentials, "_LOCK_DEADLINE_S", 0.5)
            with pytest.raises(credentials.LockAcquisitionError):
                credentials.save("cmp_test_key_test_key_1234")
        finally:
            proc.terminate()
            proc.wait(timeout=3)


class TestResolveApiKey:
    def test_explicit_wins_over_env_and_file(self, tmp_creds, monkeypatch):
        monkeypatch.setenv("COMPRESR_API_KEY", "cmp_env_XXXXXXXXXXXXX")
        credentials.save("cmp_file_test_key_1234", profile="default")
        assert (
            credentials.resolve_api_key("cmp_explicit_test_key_1234")
            == "cmp_explicit_test_key_1234"
        )

    def test_env_wins_over_file(self, tmp_creds, monkeypatch):
        monkeypatch.setenv("COMPRESR_API_KEY", "cmp_env_XXXXXXXXXXXXX")
        credentials.save("cmp_file_test_key_1234", profile="default")
        assert credentials.resolve_api_key(None) == "cmp_env_XXXXXXXXXXXXX"

    def test_file_used_when_no_explicit_no_env(self, tmp_creds):
        credentials.save("cmp_file_test_key_1234", profile="default")
        assert credentials.resolve_api_key(None) == "cmp_file_test_key_1234"

    def test_returns_none_when_nothing_configured(self, tmp_creds):
        assert credentials.resolve_api_key(None) is None

    def test_profile_env_selects_section(self, tmp_creds, monkeypatch):
        credentials.save("cmp_a_XXXXXXXXXXXXXX", profile="default")
        credentials.save("cmp_b_XXXXXXXXXXXXXX", profile="work")
        monkeypatch.setenv("COMPRESR_PROFILE", "work")
        assert credentials.resolve_api_key(None) == "cmp_b_XXXXXXXXXXXXXX"
