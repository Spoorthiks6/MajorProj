import time

import requests

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


def call_groq(api_key: str, messages: list[dict[str, str]]) -> str:
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
            print(f"Groq rate limited; retrying in {wait_seconds} seconds.")
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


def system_message(summary: str, role: str) -> dict[str, str]:
    role_instruction = ROLE_GUIDANCE[role]
    return {
        "role": "system",
        "content": (
            f"{SYSTEM_PROMPT}{summary}\n\n"
            f"Current response perspective: {role}. {role_instruction}"
        ),
    }
