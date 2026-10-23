import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from config import SESSION_SUMMARY_FILE, REPORTS_DIR

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SESSION_TEXT_FILE = REPORTS_DIR / "llm_session_summary.txt"
DASHBOARD_TEXT_FILE = REPORTS_DIR / "llm_dashboard_insights.txt"

SYSTEM_PROMPT = (
    "You are a calm, encouraging study coach. You are given structured "
    "metrics from one student's EEG-tracked study session. Give exactly "
    "3 complete sentences of practical advice tied to the student's activity "
    "and the provided metrics. Address the student by their provided name "
    "and explicitly mention the activity. Use natural, supportive language. "
    "Do not restate the input as a list or invent numbers. Focus, distraction, "
    "fatigue, and retention values are experimental heuristics, not validated "
    "measurements; never present them as facts about ability, memory, or health. "
    "Never make medical or diagnostic claims. End every sentence with a full "
    "stop, without ellipses or unfinished thoughts."
)

DASHBOARD_SYSTEM_PROMPT = (
    "You are a concise study coach. Given metrics from several recorded "
    "sessions, write exactly 3 complete sentences comparing activities and "
    "their estimated focus or retention, then recommend one practical next "
    "step. Do not address a specific student by name, invent metrics, or make "
    "medical or diagnostic claims. Treat all provided values as experimental "
    "estimates, not validated measurements."
)


GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# Return the Groq API key if configured; otherwise the script will fall back to text templates.
def _client():
    """Free hosted LLM via Groq — no local install needed, works on any
    computer with internet. Get a free key at console.groq.com and set
    GROQ_API_KEY as an environment variable."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set — get a free key at console.groq.com. Falling back to templates.")
        return None
    return api_key


# Cut off any trailing partial sentence so the note stays coherent and complete.
def _trim_to_complete_sentence(text: str) -> str:
    """Trim any incomplete tail from a model response while keeping the
    complete sentences that were already generated."""
    text = text.strip()
    if not text:
        return ""

    # Keep the last complete sentence when the model trails off mid-thought.
    for cutoff in (".", "!", "?"):
        idx = text.rfind(cutoff)
        if idx != -1:
            truncated = text[: idx + 1].strip()
            if truncated:
                return truncated
    return text


def _participant_name(value: object) -> str:
    name = str(value).strip()
    if not name or name.lower() == "unknown":
        return "Participant"
    return name.replace("_", " ").title()


# Reject blank or broken responses and use a fallback note when the LLM output is unusable.
def _repair_note(text: str, fallback: str) -> str:
    """Final safety check: reject blank, too-short, or numerically cut-off
    outputs and use a known-good fallback instead."""
    cleaned = re.sub(r"\s+", " ", _trim_to_complete_sentence(text or "")).strip()
    sentence_count = cleaned.count(".") + cleaned.count("!") + cleaned.count("?")
    if not cleaned:
        reason = "empty response"
    elif len(cleaned) < 40:
        reason = "response was too short"
    elif sentence_count < 2:
        reason = "fewer than two complete sentences"
    elif re.search(r"(?:^|\s)\d+\.$", cleaned):
        reason = "response ended with a cut-off number"
    else:
        return cleaned

    print(f"Groq response rejected ({reason}); using local template feedback, not an LLM response.")
    return fallback


# Call the Groq LLM with retries and brief rate-limit handling for generated coaching text.
def _call_llm(
    client,
    prompt: str,
    max_retries: int = 3,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    import requests
    import time

    for attempt in range(max_retries):
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {client}"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 500,
                "temperature": 0.2,
                "top_p": 0.9,
            },
            timeout=45,
        )

        if response.status_code == 429:
            if attempt + 1 < max_retries:
                wait_s = 16  # Groq free tier resets per-minute; a bit over 15s to be safe
                print(f"  Rate limited — waiting {wait_s}s before retry ({attempt + 1}/{max_retries})...")
                time.sleep(wait_s)
            continue

        response.raise_for_status()
        try:
            result = response.json()
        except ValueError as error:
            raise RuntimeError("Groq returned an invalid JSON response.") from error

        if not isinstance(result, dict):
            raise RuntimeError("Groq returned a response in an unexpected format.")
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RuntimeError("Groq returned no completion choices.")

        message = choices[0].get("message")
        text = message.get("content") if isinstance(message, dict) else None
        text = text.strip() if isinstance(text, str) else ""
        if not text:
            if attempt + 1 < max_retries:
                print(f"  Groq returned an empty response — retrying ({attempt + 1}/{max_retries})...")
            continue
        return _trim_to_complete_sentence(text)

    raise RuntimeError(
        f"Groq returned no usable response after {max_retries} attempts "
        "(empty response or rate limit)."
    )


# Build a plain-language summary when no LLM response is available for a session.
def fallback_session_text(row: pd.Series) -> str:
    return (
        f"Your estimated deep-focus time was {row['deep_focus_min']:.1f} of "
        f"{row['duration_min']:.1f} minutes; try starting your next "
        f"{row['activity']} session with one short, single-task study block and "
        "extend it gradually if it feels manageable. "
        f"The distraction heuristic flagged {row['distraction_min']:.1f} minutes, "
        f"so reduce interruptions during {row['activity']} by silencing "
        "notifications and keeping only the needed materials in view. "
        f"Your session received a {row['mental_fatigue'].lower()} fatigue label "
        f"from an experimental rule, so treat the recommended "
        f"{row['recommended_break_min']}-minute break and "
        f"{row['tomorrow_suggested_duration_min']}-minute next-session duration "
        "as rough suggestions, not validated targets."
    )


def _generate_note(
    client: str,
    prompt: str,
    fallback: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[str, bool]:
    import requests

    try:
        response_text = _call_llm(client, prompt, system_prompt=system_prompt)
    except (requests.RequestException, RuntimeError) as error:
        print(
            f"Groq feedback unavailable ({error}); using local template feedback, "
            "not an LLM response."
        )
        return fallback, True

    note = _repair_note(response_text, fallback)
    return note, note == fallback


# Generate per-session coaching notes and a summary insight from the session metrics.
def generate_narratives(summary_df: pd.DataFrame):
    import time
    client = _client()

    session_lines = []
    for _, row in summary_df.iterrows():
        fallback = fallback_session_text(row)
        person = _participant_name(row.get("person", "unknown"))
        activity = str(row["activity"]).replace("_", " ").strip()
        if client:
            prompt = (
                f"Student name: {person}\n"
                f"Activity: {activity}\n"
                f"Session metrics:\n{row.to_json()}\n\n"
                f"Write exactly 3 complete sentences of useful, specific "
                f"feedback for {person} about this {activity} session. "
                "Use the name and activity naturally in the advice. Base "
                "suggestions only on these metrics, and describe heuristic "
                "values as estimates rather than validated measurements."
            )
            text, used_fallback = _generate_note(client, prompt, fallback)
            time.sleep(2)  # spread out calls to stay under free-tier tokens-per-minute limit
        else:
            text = fallback
            used_fallback = True
        source = "template fallback" if used_fallback else "LLM-generated"
        session_lines.append(f"{person} — {activity} [{source}]\n{text}")

    if client:
        summary_cols = ["activity", "duration_min", "mental_fatigue",
                         "predicted_retention_pct", "recommended_break_min"]
        dashboard_prompt = (
            "Here are metrics for all sessions this student has recorded:\n"
            + summary_df[summary_cols].to_json(orient="records")
            + "\n\nWrite exactly 3 complete sentences comparing "
              "which activities were associated with higher focus/retention "
              "and what the student should try next."
        )
        highest = summary_df.sort_values("predicted_retention_pct", ascending=False).iloc[0]
        lowest = summary_df.sort_values("predicted_retention_pct").iloc[0]
        dashboard_fallback = (
            f"Across {len(summary_df)} sessions, '{highest['activity']}' showed the highest "
            f"estimated retention ({highest['predicted_retention_pct']:.0f}%), while "
            f"'{lowest['activity']}' showed the lowest "
            f"({lowest['predicted_retention_pct']:.0f}%). "
            "This is a local heuristic summary, not an LLM response."
        )
        dashboard_text, used_fallback = _generate_note(
            client,
            dashboard_prompt,
            dashboard_fallback,
            system_prompt=DASHBOARD_SYSTEM_PROMPT,
        )
        source = "Template fallback (not LLM-generated)" if used_fallback else "LLM-generated"
        dashboard_text = f"{source}: {dashboard_text}"
    else:
        top = summary_df.sort_values("predicted_retention_pct", ascending=False).iloc[0]
        low = summary_df.sort_values("predicted_retention_pct").iloc[0]
        dashboard_text = (
            f"Across {len(summary_df)} sessions, '{top['activity']}' showed the highest "
            f"estimated retention ({top['predicted_retention_pct']:.0f}%), while "
            f"'{low['activity']}' showed the lowest ({low['predicted_retention_pct']:.0f}%). "
            "This is a local heuristic summary, not an LLM response."
        )
        dashboard_text = f"Template fallback (not LLM-generated): {dashboard_text}"

    return "\n\n".join(session_lines), dashboard_text


# Save the final session notes and dashboard insight text files for the report builder.
def main() -> None:
    if not SESSION_SUMMARY_FILE.exists():
        raise FileNotFoundError("Run session_report.py first.")

    df = pd.read_csv(SESSION_SUMMARY_FILE)
    session_text, dashboard_text = generate_narratives(df)

    SESSION_TEXT_FILE.write_text(session_text, encoding="utf-8")
    DASHBOARD_TEXT_FILE.write_text(dashboard_text, encoding="utf-8")

    print(f"Saved: {SESSION_TEXT_FILE}")
    print(f"Saved: {DASHBOARD_TEXT_FILE}")


if __name__ == "__main__":
    main()
