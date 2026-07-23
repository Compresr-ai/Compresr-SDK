"""``compresr-sdk`` CLI: login, logout, whoami, status."""

from __future__ import annotations

import argparse
import configparser
import sys
from typing import Optional

from . import __version__
from .auth import login as _login
from .auth import logout as _logout
from .credentials import (
    DEFAULT_PROFILE,
    LockAcquisitionError,
    credentials_path,
    load,
)


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _cmd_login(args: argparse.Namespace) -> int:
    try:
        _login(
            app_url=args.app_url,
            base_url=args.base_url,
            profile=args.profile,
            timeout=args.timeout,
            open_browser=not args.no_browser,
        )
        return 0
    except (RuntimeError, TimeoutError, ValueError) as e:
        _err(f"error: {e}")
        return 1
    except PermissionError as e:
        _err(f"error: {e}")
        _err("hint: run `chmod 600 ~/.compresr/credentials` and retry.")
        return 1


def _cmd_logout(args: argparse.Namespace) -> int:
    try:
        removed = _logout(
            args.profile,
            revoke_server_key=not args.no_server_revoke,
            base_url=args.base_url,
        )
    except LockAcquisitionError as e:
        _err(f"error: {e}")
        return 1
    except PermissionError as e:
        _err(f"error: {e}")
        _err("hint: run `chmod 600 ~/.compresr/credentials` and retry.")
        return 1
    if removed:
        print(f"Removed credentials for profile '{args.profile}'.")
        return 0
    print(f"No credentials found for profile '{args.profile}'.")
    return 0


def _cmd_whoami(args: argparse.Namespace) -> int:
    try:
        key = load(args.profile)
    except PermissionError as e:
        _err(f"error: {e}")
        return 1
    if not key:
        _err(f"Not logged in (profile '{args.profile}'). Run `compresr-sdk login`.")
        return 1
    print(f"profile:  {args.profile}")
    print(f"file:     {credentials_path()}")
    print(f"api_key:  {key[:8]}…{key[-4:]}")
    return 0


def _cmd_status(_args: argparse.Namespace) -> int:
    from .credentials import _read_file

    path = credentials_path()
    print(f"credentials file: {path}")
    print(f"exists:           {path.exists()}")
    if not path.exists():
        return 0
    try:
        parser = _read_file()
    except PermissionError as e:
        _err(f"error: {e}")
        return 1
    except configparser.Error as e:
        _err(f"warning: file is corrupt ({e})")
        return 0
    profiles = parser.sections()
    print(f"profiles:         {', '.join(profiles) if profiles else '(none)'}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="compresr-sdk", description="Compresr SDK CLI")
    p.add_argument("--version", action="version", version=f"compresr-sdk {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    login_p = sub.add_parser("login", help="Log in via browser and store credentials")
    login_p.add_argument("--profile", default=DEFAULT_PROFILE)
    login_p.add_argument("--app-url", default=None, help="Override web app URL")
    login_p.add_argument("--base-url", default=None, help="Override API base URL")
    login_p.add_argument("--timeout", type=float, default=120.0)
    login_p.add_argument(
        "--no-browser", action="store_true", help="Print the URL instead of opening it"
    )
    login_p.set_defaults(func=_cmd_login)

    logout_p = sub.add_parser(
        "logout",
        help="Revoke the server-side CLI key and remove stored credentials",
    )
    logout_p.add_argument("--profile", default=DEFAULT_PROFILE)
    logout_p.add_argument("--base-url", default=None, help="Override API base URL")
    logout_p.add_argument(
        "--no-server-revoke",
        action="store_true",
        help="Skip the server-side key revoke and only delete local credentials.",
    )
    logout_p.set_defaults(func=_cmd_logout)

    whoami_p = sub.add_parser("whoami", help="Show the stored API key preview")
    whoami_p.add_argument("--profile", default=DEFAULT_PROFILE)
    whoami_p.set_defaults(func=_cmd_whoami)

    status_p = sub.add_parser("status", help="Show credentials file location and profiles")
    status_p.set_defaults(func=_cmd_status)

    return p


def main(argv: Optional[list] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        _err("\nAborted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
