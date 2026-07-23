"""Tests for the packaged shim installer.

The installer normally writes into the active litellm install. Here we redirect
its target to a tmp dir so tests don't mutate site-packages.
"""

from pathlib import Path

import pytest

pytest.importorskip("litellm")

from compresr.integrations.litellm import shim_installer  # noqa: E402


@pytest.fixture
def fake_hooks_dir(tmp_path, monkeypatch):
    """Redirect the installer's target to a tmp dir."""
    hooks_dir = tmp_path / "guardrail_hooks"
    hooks_dir.mkdir()
    monkeypatch.setattr(shim_installer, "_hooks_dir", lambda: hooks_dir)
    return hooks_dir


def test_install_writes_shim_and_is_idempotent(fake_hooks_dir, capsys):
    target = fake_hooks_dir / "compresr" / "__init__.py"
    assert not target.exists()

    assert shim_installer.install(verbose=False) is True
    assert target.exists()
    assert "guardrail_initializer_registry" in target.read_text()
    assert "from compresr.integrations.litellm import" in target.read_text()

    # Second install: no-op, returns False, doesn't overwrite.
    original = target.read_text()
    assert shim_installer.install(verbose=False) is False
    assert target.read_text() == original


def test_uninstall_removes_shim(fake_hooks_dir):
    shim_installer.install(verbose=False)
    target = fake_hooks_dir / "compresr" / "__init__.py"
    assert target.exists()

    assert shim_installer.uninstall(verbose=False) is True
    assert not target.exists()

    # Uninstall when missing: no-op, returns False.
    assert shim_installer.uninstall(verbose=False) is False


def test_is_installed_reflects_target_state(fake_hooks_dir):
    assert shim_installer.is_installed() is False
    shim_installer.install(verbose=False)
    assert shim_installer.is_installed() is True
    shim_installer.uninstall(verbose=False)
    assert shim_installer.is_installed() is False


def test_shim_content_is_importable_python():
    """The shipped SHIM_CONTENT must be syntactically valid Python and
    reference only public exports of the integration package."""
    compile(shim_installer.SHIM_CONTENT, "<shim>", "exec")
    assert "guardrail_initializer_registry" in shim_installer.SHIM_CONTENT
    assert "guardrail_class_registry" in shim_installer.SHIM_CONTENT
    assert "initialize_guardrail" in shim_installer.SHIM_CONTENT


def test_main_install_and_uninstall(fake_hooks_dir, monkeypatch, capsys):
    target = fake_hooks_dir / "compresr" / "__init__.py"

    monkeypatch.setattr("sys.argv", ["install-compresr-shim"])
    assert shim_installer.main() == 0
    assert target.exists()

    monkeypatch.setattr("sys.argv", ["install-compresr-shim", "--uninstall"])
    assert shim_installer.main() == 0
    assert not target.exists()


def test_auto_install_via_env_var(fake_hooks_dir, monkeypatch):
    """Importing the integration package with COMPRESR_AUTO_INSTALL_SHIM=1
    triggers shim install. Without the env var, import is a no-op."""
    target = fake_hooks_dir / "compresr" / "__init__.py"
    assert not target.exists()

    monkeypatch.setenv("COMPRESR_AUTO_INSTALL_SHIM", "1")
    from compresr.integrations.litellm import _maybe_auto_install_shim

    _maybe_auto_install_shim()
    assert target.exists()
    target.unlink()
    target.parent.rmdir()

    monkeypatch.delenv("COMPRESR_AUTO_INSTALL_SHIM", raising=False)
    _maybe_auto_install_shim()
    assert not target.exists()


def test_path_helpers_match(fake_hooks_dir):
    """_target_init resolves to the expected location under _hooks_dir."""
    expected = fake_hooks_dir / "compresr" / "__init__.py"
    assert shim_installer._target_init() == expected
    assert isinstance(shim_installer._hooks_dir(), Path)
