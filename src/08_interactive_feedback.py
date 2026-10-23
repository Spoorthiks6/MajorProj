import os
import subprocess
import sys
import time
from pathlib import Path

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

from config import SESSION_SUMMARY_FILE

load_dotenv(PROJECT_ROOT / ".env")

GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_RETRIES = 3
MAX_HISTORY_MESSAGES = 12

ROLE_GUIDANCE = {
    "student": "Speak directly and encouragingly to the student. Give practical steps they can try.",
    "teacher": "Speak to a teacher supporting the learner. Suggest supportive, non-judgmental ways to guide them.",
    "parent": "Speak to a parent or caregiver. Offer practical, supportive ways to encourage the learner.",
    "coach": "Speak as a study coach. Give specific, realistic coaching suggestions.",
    "overall": "Give an overall view across the recorded sessions, including useful patterns and next steps.",
}

SYSTEM_PROMPT = (
    "You are an interactive study-feedback assistant. Answer the user's question "
    "using only the session summary included below. Use the requested audience "
    "perspective and answer follow-up questions using the conversation. Be clear, "
    "supportive, and practical. Compare activities or sessions only when the data "
    "supports it. If the summary does not contain enough information, say so "
    "instead of guessing. Focus, distraction, fatigue, and retention fields are "
    "experimental heuristics, not validated measurements; never present them as "
    "facts about ability, memory, or health. Do not make medical or diagnostic "
    "claims.\n\nSession summary (JSON):\n"
)


def _call_groq(api_key: str, messages: list[dict[str, str]]) -> str:
    for attempt in range(MAX_RETRIES):
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.2,
            },
            timeout=45,
        )

        if response.status_code == 429 and attempt + 1 < MAX_RETRIES:
            wait_seconds = 16
            print(f"Rate limited; retrying in {wait_seconds} seconds.")
            time.sleep(wait_seconds)
            continue

        if response.status_code == 429:
            raise RuntimeError("Groq is still rate-limiting requests. Wait a minute and try again.")

        response.raise_for_status()
        try:
            result = response.json()
        except ValueError as error:
            raise RuntimeError("Groq returned an invalid JSON response.") from error

        if not isinstance(result, dict):
            raise RuntimeError("Groq returned a response in an unexpected format.")
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RuntimeError(f"Groq returned no answer: {result}")

        message = choices[0].get("message")
        answer = message.get("content", "") if isinstance(message, dict) else ""
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Groq returned an empty answer.")
        return answer.strip()

    raise RuntimeError("Could not get a response from Groq.")


def _system_message(summary: str, role: str) -> dict[str, str]:
    role_instruction = ROLE_GUIDANCE[role]
    return {
        "role": "system",
        "content": (
            f"{SYSTEM_PROMPT}{summary}\n\n"
            f"Current response perspective: {role}. {role_instruction}"
        ),
    }


def run_chat(summary_df: pd.DataFrame, api_key: str) -> None:
    relevant_columns = [
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
    available_columns = [column for column in relevant_columns if column in summary_df.columns]
    summary = summary_df[available_columns].to_json(orient="records", indent=2)

    print("Interactive study feedback")
    print("Perspectives: student, teacher, parent, coach, overall")
    print("Commands: /role <perspective>, /overall, /clear, /help, /quit")
    print("Type a question to discuss session feedback or ask a follow-up.\n")

    role = "student"
    history: list[dict[str, str]] = []
    while True:
        try:
            question = input(f"{role}> ").strip()
        except EOFError:
            print()
            return

        if not question:
            continue
        if question in {"/quit", "/exit"}:
            return
        if question == "/help":
            print("Ask any question about the recorded sessions. Use /role student, "
                  "teacher, parent, coach, or overall to change perspective.")
            print("/overall gives a summary across the recorded sessions; /clear starts a fresh conversation.")
            continue
        if question == "/clear":
            history.clear()
            print("Conversation cleared.")
            continue
        if question == "/overall":
            role = "overall"
            question = "Give me an overall feedback summary across the recorded sessions and suggest practical next steps."
        elif question.startswith("/role"):
            requested_role = question.removeprefix("/role").strip().lower()
            if requested_role not in ROLE_GUIDANCE:
                print("Choose one: student, teacher, parent, coach, or overall.")
                continue
            role = requested_role
            print(f"Perspective set to {role}.")
            continue

        messages = [_system_message(summary, role), *history[-MAX_HISTORY_MESSAGES:]]
        messages.append({"role": "user", "content": question})
        try:
            answer = _call_groq(api_key, messages)
        except requests.RequestException as error:
            print(f"Request to Groq failed: {error}")
            continue
        except RuntimeError as error:
            print(f"Could not generate feedback: {error}")
            continue

        print(f"\n{answer}\n")
        history.extend(({"role": "user", "content": question}, {"role": "assistant", "content": answer}))


def main() -> int:
    if not SESSION_SUMMARY_FILE.exists():
        print("Session summary not found. Run `python3 run_pipeline.py` first.", file=sys.stderr)
        return 1

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print(
            "GROQ_API_KEY is required for interactive feedback. Set it in the project .env "
            "file or your environment, then try again.",
            file=sys.stderr,
        )
        return 1

    summary_df = pd.read_csv(SESSION_SUMMARY_FILE)
    if summary_df.empty:
        print("The session summary contains no sessions to discuss.", file=sys.stderr)
        return 1

    run_chat(summary_df, api_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
