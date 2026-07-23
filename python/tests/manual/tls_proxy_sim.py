"""Hermetic simulation of a TLS-intercepting corporate proxy.

Reproduces — and previews the fix for — the agents "Agent execution failed:
Connection error" failure seen on a customer machine whose corporate proxy
intercepts TLS to ``api.anthropic.com`` with a CA that Python's ``certifi``
bundle does not trust.

No real network, no real API key, no mitmproxy: we stand up a local HTTPS
server with a *self-signed* cert (the stand-in for the proxy's untrusted CA),
redirect the Anthropic SDK at it via ``ANTHROPIC_BASE_URL``, and drive the
REAL ``_Engine`` so the whole ``create_agent -> ChatAnthropic -> anthropic
SDK -> httpx -> ssl`` stack runs.

Scenarios
---------
A. MITM / untrusted cert  -> default verify=True  -> expect cert failure
B. Fix preview            -> trust the test CA    -> expect the call to land
C. Egress block           -> closed port          -> expect ConnectError

Run:  python tests/manual/tls_proxy_sim.py
"""

from __future__ import annotations

import datetime
import http.server
import ipaddress
import json
import os
import ssl
import tempfile
import threading
from typing import Any

import httpx
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

CN = "Acme Corp Inspection Proxy"


def _make_self_signed(tmpdir: str) -> tuple[str, str]:
    """Write a self-signed cert+key for 127.0.0.1/localhost; return their paths.

    This cert is signed by an authority Python has never heard of — exactly
    like a corporate TLS-interception proxy's certificate.
    """
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, CN)])
    san = x509.SubjectAlternativeName(
        [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(san, critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path = os.path.join(tmpdir, "cert.pem")
    key_path = os.path.join(tmpdir, "key.pem")
    with open(cert_path, "wb") as fh:
        fh.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(key_path, "wb") as fh:
        fh.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            )
        )
    return cert_path, key_path


class _FakeAnthropic(http.server.BaseHTTPRequestHandler):
    """Returns a minimal Anthropic Messages response and flags that it was hit.

    If a request reaches here, the TLS handshake SUCCEEDED — so ``server.hit``
    is the discriminator between 'cert rejected' (A) and 'cert trusted' (B).
    """

    server_version = "FakeAnthropic/0"

    def log_message(self, *a: Any) -> None:  # silence
        pass

    def do_POST(self) -> None:
        self.server.hit = True  # type: ignore[attr-defined]
        body = json.dumps(
            {
                "id": "msg_sim",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "hi from the fake proxy"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _start_server(cert_path: str, key_path: str) -> http.server.HTTPServer:
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _FakeAnthropic)
    httpd.hit = False  # type: ignore[attr-defined]
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


class _StubClient:
    """Minimal stand-in for CompressionClient (no tools => compress unused)."""

    def compress(self, **kw: Any) -> Any:  # pragma: no cover - not hit here
        raise AssertionError("compress should not be called in this sim")


def _run_engine_once(**engine_kwargs: Any) -> Any:
    """Drive the real ``_Engine`` once. ``engine_kwargs`` is forwarded to the
    constructor so a caller can set ``llm_http_client`` / ``llm_http_async_client``
    once, the same way you'd pass ``http_client`` to ``Anthropic(...)``."""
    from compresr.agents.engine import _Engine

    engine = _Engine(
        compresr_client=_StubClient(),
        llm="anthropic",
        llm_api_key="sk-ant-simulated-key",
        **engine_kwargs,
    )
    return engine.run(
        messages=[{"role": "user", "content": "hi"}],
        model="claude-sonnet-4-6",
        max_tokens=8,
    )


def scenario_a_untrusted(server: http.server.HTTPServer, port: int) -> None:
    print("\n=== Scenario A: TLS-intercepting proxy, default verify=True ===")
    os.environ["ANTHROPIC_BASE_URL"] = f"https://127.0.0.1:{port}"
    server.hit = False  # type: ignore[attr-defined]
    try:
        out = _run_engine_once()
        print("  UNEXPECTED SUCCESS:", repr(out)[:120])
    except Exception as exc:  # noqa: BLE001
        cause = exc.__cause__
        print("  raised:", type(exc).__name__, "->", str(exc)[:140])
        print(
            "  underlying cause:", type(cause).__name__ if cause else None, "->", str(cause)[:140]
        )
        print("  server_was_hit:", getattr(server, "hit", None), "(False = handshake rejected)")


def scenario_b_http_client_verify_false(server: http.server.HTTPServer, port: int) -> None:
    print("\n=== Scenario B: same proxy, http_client=verify=False (the fix) ===")
    os.environ["ANTHROPIC_BASE_URL"] = f"https://127.0.0.1:{port}"
    server.hit = False  # type: ignore[attr-defined]

    # The fix: hand the client an httpx client with TLS verification disabled,
    # once at construction (CompressionClient(..., llm_http_client=...)). It rides
    # through init_chat_model -> ChatAnthropic(http_client=...) -> the anthropic
    # SDK, so the intercepting cert is no longer rejected. verify=False is the
    # blunt corporate-proxy escape hatch; verify="/path/corp-ca.pem" is the safer
    # real-world form.
    sync_client = httpx.Client(verify=False, timeout=5)
    async_client = httpx.AsyncClient(verify=False, timeout=5)
    try:
        out = _run_engine_once(llm_http_client=sync_client, llm_http_async_client=async_client)
        print("  call landed. result:", repr(out)[:120])
    except Exception as exc:  # noqa: BLE001
        print("  raised:", type(exc).__name__, "->", str(exc)[:160])
    finally:
        sync_client.close()
        print("  server_was_hit:", getattr(server, "hit", None), "(True = handshake trusted)")


def scenario_c_egress_block() -> None:
    print("\n=== Scenario C: egress block (host not reachable) ===")
    os.environ["ANTHROPIC_BASE_URL"] = "https://127.0.0.1:9"  # discard port, closed
    try:
        out = _run_engine_once()
        print("  UNEXPECTED SUCCESS:", repr(out)[:120])
    except Exception as exc:  # noqa: BLE001
        cause = exc.__cause__
        print("  raised:", type(exc).__name__, "->", str(exc)[:140])
        print("  underlying cause:", type(cause).__name__ if cause else None)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cert_path, key_path = _make_self_signed(tmp)
        server = _start_server(cert_path, key_path)
        port = server.socket.getsockname()[1]
        print(f"Local fake-Anthropic TLS server on 127.0.0.1:{port} (CN={CN!r})")
        try:
            scenario_a_untrusted(server, port)
            scenario_b_http_client_verify_false(server, port)
            scenario_c_egress_block()
        finally:
            server.shutdown()
            os.environ.pop("ANTHROPIC_BASE_URL", None)


if __name__ == "__main__":
    main()
