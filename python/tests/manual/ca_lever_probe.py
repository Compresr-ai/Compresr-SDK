"""Find which CA-injection lever actually makes the Anthropic stack trust a
custom CA, given ChatAnthropic does not declare an ``http_client`` field.

Reuses the self-signed local server from tls_proxy_sim. Tests levers from
cleanest (env var) to most invasive, at both the raw-httpx and ChatAnthropic
layers, and reports which ones let the handshake land.
"""

from __future__ import annotations

import os
import tempfile

import httpx
from tls_proxy_sim import _make_self_signed, _start_server


def _probe(label: str, fn) -> None:
    try:
        fn()
        print(f"  [LANDS]  {label}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [fails]  {label}: {type(exc).__name__}: {str(exc)[:80]}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cert_path, key_path = _make_self_signed(tmp)
        server = _start_server(cert_path, key_path)
        port = server.socket.getsockname()[1]
        url = f"https://127.0.0.1:{port}"
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        }
        print(f"server on {url}\n--- raw httpx ---")

        _probe(
            "httpx default verify (control: should fail)", lambda: httpx.Client(timeout=5).post(url)
        )
        _probe("httpx verify=ca_path", lambda: httpx.Client(verify=cert_path, timeout=5).post(url))

        os.environ["SSL_CERT_FILE"] = cert_path
        _probe("SSL_CERT_FILE env + httpx default", lambda: httpx.Client(timeout=5).post(url))
        os.environ.pop("SSL_CERT_FILE", None)

        print("--- anthropic SDK ---")
        import anthropic

        def _anthropic_http_client() -> None:
            c = anthropic.Anthropic(
                api_key="sk-sim",
                base_url=url,
                http_client=httpx.Client(verify=cert_path, timeout=5),
            )
            c.messages.create(**body)

        _probe("anthropic.Anthropic(http_client=verify=ca)", _anthropic_http_client)

        os.environ["SSL_CERT_FILE"] = cert_path
        _probe(
            "anthropic.Anthropic(base_url) + SSL_CERT_FILE env",
            lambda: anthropic.Anthropic(api_key="sk-sim", base_url=url).messages.create(**body),
        )
        os.environ.pop("SSL_CERT_FILE", None)

        print("--- ChatAnthropic (does it accept http_client despite no field?) ---")
        from langchain_anthropic import ChatAnthropic

        def _chat_http_client_ca() -> None:
            chat = ChatAnthropic(
                model="claude-sonnet-4-6",
                api_key="sk-sim",
                anthropic_api_url=url,
                max_tokens=8,
                http_client=httpx.Client(verify=cert_path, timeout=5),
            )
            chat.invoke("hi")

        def _chat_http_client_no_verify() -> None:
            chat = ChatAnthropic(
                model="claude-sonnet-4-6",
                api_key="sk-sim",
                anthropic_api_url=url,
                max_tokens=8,
                http_client=httpx.Client(verify=False, timeout=5),
            )
            chat.invoke("hi")

        _probe("ChatAnthropic(http_client=verify=ca)", _chat_http_client_ca)
        _probe("ChatAnthropic(http_client=verify=False)", _chat_http_client_no_verify)

        os.environ["SSL_CERT_FILE"] = cert_path
        _probe(
            "ChatAnthropic(anthropic_api_url) + SSL_CERT_FILE env",
            lambda: ChatAnthropic(
                model="claude-sonnet-4-6", api_key="sk-sim", anthropic_api_url=url, max_tokens=8
            ).invoke("hi"),
        )
        os.environ.pop("SSL_CERT_FILE", None)

        # truststore: uses the OS store, which does NOT contain our test CA, so
        # we expect it to still fail — documents that truststore can't be
        # validated against an arbitrary test CA (only OS-installed ones).
        try:
            import truststore  # type: ignore

            truststore.inject_into_ssl()
            _probe(
                "truststore.inject_into_ssl + httpx default (OS store only)",
                lambda: httpx.Client(timeout=5).post(url),
            )
        except ImportError:
            print("  [skip ]  truststore not installed")

        server.shutdown()


if __name__ == "__main__":
    main()
