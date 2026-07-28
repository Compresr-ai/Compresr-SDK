"""Unit tests for the Hermes recovery cache (fake Hermes runtime)."""

import os
import time

import pytest

from .hermes_fakes import FakeLocalEnvironment


@pytest.fixture
def cache_mod(hermes_env):
    return hermes_env.load("cache")


class TestStoreOriginal:
    def test_local_roundtrip(self, hermes_env, cache_mod):
        path = cache_mod.store_original("abc123", "original bytes")
        assert path is not None
        assert path == str(cache_mod.cache_file_path("abc123").resolve())
        with open(path, encoding="utf-8") as f:
            assert f.read() == "original bytes"

    def test_restrictive_file_mode(self, cache_mod):
        path = cache_mod.store_original("perm1", "content")
        assert oct(os.stat(path).st_mode & 0o777) == "0o600"

    def test_write_failure_returns_none(self, cache_mod, monkeypatch):
        monkeypatch.setattr(os, "replace", _raise_oserror)
        assert cache_mod.store_original("fail1", "content") is None

    def test_identical_content_dedupes_onto_same_path(self, cache_mod):
        p1 = cache_mod.store_original("same", "content")
        p2 = cache_mod.store_original("same", "content")
        assert p1 == p2

    def test_local_environment_class_uses_host_path(self, hermes_env, cache_mod):
        hermes_env.reinstall(active_env=FakeLocalEnvironment())
        cache_mod = hermes_env.load("cache")
        path = cache_mod.store_original("env1", "content")
        assert path == str(cache_mod.cache_file_path("env1").resolve())

    def test_unknown_backend_fails_open(self, hermes_env, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "kubernetes")
        cache_mod = hermes_env.load("cache")
        assert cache_mod.store_original("env2", "content") is None
        assert cache_mod.cache_file_path("env2").exists()


class TestPrune:
    def test_prune_evicts_oldest_beyond_limit(self, cache_mod):
        root = cache_mod.ensure_cache_root()
        old = time.time() - 3600
        for i in range(5):
            p = root / f"entry{i}"
            p.write_text("x" * 1024)
            os.utime(p, (old + i, old + i))
        keep = root / "keep"
        keep.write_text("x" * 1024)
        cache_mod._prune_cache_dir(str(root), 3 * 1024, str(keep))
        remaining = sorted(p.name for p in root.iterdir())
        assert "keep" in remaining
        assert "entry0" not in remaining
        assert len(remaining) <= 4

    def test_recent_entries_pinned(self, cache_mod):
        root = cache_mod.ensure_cache_root()
        for i in range(3):
            (root / f"fresh{i}").write_text("x" * 1024)
        keep = root / "keep"
        keep.write_text("x")
        cache_mod._prune_cache_dir(str(root), 1024, str(keep))
        assert sorted(p.name for p in root.iterdir()) == ["fresh0", "fresh1", "fresh2", "keep"]

    def test_zero_budget_noop(self, cache_mod):
        root = cache_mod.ensure_cache_root()
        (root / "a").write_text("x")
        cache_mod._prune_cache_dir(str(root), 0, str(root / "a"))
        assert (root / "a").exists()


def _raise_oserror(*args, **kwargs):
    raise OSError("disk full")
