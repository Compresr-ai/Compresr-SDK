#!/usr/bin/env python3
"""
Tiny OpenAI-compatible "echo" upstream for the proxy demo.

It accepts POST /v1/chat/completions, records the EXACT request body the
LiteLLM proxy forwarded (so we can prove the Compresr guardrail rewrote the
tool output), and returns a canned, valid ChatCompletion response.

This stands in for OpenAI/Anthropic so the demo needs no upstream API key.
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

RECORD_PATH = sys.argv[2] if len(sys.argv) > 2 else "forwarded_request.json"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8199


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):  # silence default stderr noise
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"_raw": raw.decode("utf-8", "replace")}

        # Persist exactly what the proxy forwarded upstream.
        with open(RECORD_PATH, "w") as f:
            json.dump(payload, f, indent=2)

        response = {
            "id": "chatcmpl-echo",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", "echo-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "[echo upstream] received and recorded the forwarded prompt.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"echo upstream listening on http://127.0.0.1:{PORT}/v1 -> recording to {RECORD_PATH}")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
