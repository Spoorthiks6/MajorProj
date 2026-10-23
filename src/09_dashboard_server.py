import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

if VENV_DIR.exists() and Path(sys.prefix).resolve() != VENV_DIR.resolve():
    if not VENV_PYTHON.is_file():
        raise FileNotFoundError(
            f"Found {VENV_DIR}, but its Python interpreter is missing: {VENV_PYTHON}"
        )
    result = subprocess.run([str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
    raise SystemExit(result.returncode)

import pandas as pd
import requests
from dotenv import load_dotenv

from config import DASHBOARD_HTML_FILE, DASHBOARD_IMAGE_FILE, SESSION_SUMMARY_FILE
from feedback_chat import MAX_HISTORY_MESSAGES, ROLE_GUIDANCE, call_groq, system_message

load_dotenv(PROJECT_ROOT / ".env")
HOST = "127.0.0.1"
PORT = 8765
MAX_REQUEST_BYTES = 32_768
MAX_MESSAGE_LENGTH = 1_500
SUMMARY_COLUMNS = [
    "person",
    "activity",
    "duration_min",
    "deep_focus_min",
    "distraction_min",
    "avg_attention",
    "avg_meditation",
    "mental_fatigue",
    "predicted_retention_pct",
    "recommended_break_min",
    "tomorrow_suggested_duration_min",
]


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "NeuroStudyDashboard/1.0"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, str]) -> None:
        self._send(
            status,
            json.dumps(payload).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _host_is_local(self) -> bool:
        hostname = urlsplit(f"//{self.headers.get('Host', '')}").hostname
        return hostname in {"localhost", "127.0.0.1"}

    def do_GET(self) -> None:
        if not self._host_is_local():
            self._send_json(403, {"error": "Only local browser requests are allowed."})
            return
        if self.path in {"/", "/index.html"}:
            path = DASHBOARD_HTML_FILE
            content_type = "text/html; charset=utf-8"
        elif self.path == "/dashboard_summary.png":
            path = DASHBOARD_IMAGE_FILE
            content_type = "image/png"
        else:
            self._send_json(404, {"error": "Not found."})
            return

        if not path.is_file():
            self._send_json(404, {"error": "Dashboard file not found. Run the pipeline first."})
            return
        self._send(200, path.read_bytes(), content_type)

    def do_POST(self) -> None:
        if not self._host_is_local():
            self._send_json(403, {"error": "Only local browser requests are allowed."})
            return
        if self.path != "/api/chat":
            self._send_json(404, {"error": "Not found."})
            return
        if self.headers.get_content_type() != "application/json":
            self._send_json(415, {"error": "Send the question as JSON."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid request length."})
            return
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "The conversation is too large. Start a new conversation."})
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Invalid JSON request."})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "Invalid chat request."})
            return

        role = payload.get("role")
        messages = payload.get("messages")
        if role not in ROLE_GUIDANCE:
            self._send_json(400, {"error": "Choose a valid feedback perspective."})
            return
        if not isinstance(messages, list) or not messages or len(messages) > MAX_HISTORY_MESSAGES:
            self._send_json(400, {"error": "Invalid conversation history. Start a new conversation."})
            return

        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
                or len(message["content"]) > MAX_MESSAGE_LENGTH
            ):
                self._send_json(400, {"error": "Invalid conversation message."})
                return
        if messages[-1]["role"] != "user":
            self._send_json(400, {"error": "The latest conversation message must be a question."})
            return

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            self._send_json(503, {"error": "GROQ_API_KEY is not configured on the server."})
            return
        if not SESSION_SUMMARY_FILE.is_file():
            self._send_json(503, {"error": "Session summary not found. Run the pipeline first."})
            return

        try:
            summary_df = pd.read_csv(SESSION_SUMMARY_FILE)
            available_columns = [
                column for column in SUMMARY_COLUMNS if column in summary_df.columns
            ]
            summary = summary_df[available_columns].to_json(orient="records", indent=2)
            answer = call_groq(
                api_key,
                [system_message(summary, role), *messages[-MAX_HISTORY_MESSAGES:]],
            )
        except (OSError, pd.errors.ParserError) as error:
            print(f"Could not load session summary: {error}", file=sys.stderr)
            self._send_json(500, {"error": "Could not read the session summary."})
            return
        except requests.RequestException as error:
            print(f"Groq request failed: {error}", file=sys.stderr)
            self._send_json(502, {"error": "Groq could not be reached. Please try again."})
            return
        except RuntimeError as error:
            print(f"Groq feedback failed: {error}", file=sys.stderr)
            self._send_json(502, {"error": str(error)})
            return

        self._send_json(200, {"answer": answer})

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    if not DASHBOARD_HTML_FILE.is_file():
        raise FileNotFoundError("Dashboard not found. Run `python3 run_pipeline.py` first.")
    if not SESSION_SUMMARY_FILE.is_file():
        raise FileNotFoundError("Session summary not found. Run `python3 run_pipeline.py` first.")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Set GROQ_API_KEY in the project .env file before starting the dashboard.")

    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"Live dashboard: http://localhost:{PORT}/")
    print("Keep this terminal open while chatting. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the NeuroStudy dashboard server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
