"""
query_engine.py – Natural-language query interpreter for Smart Desk Agent.

All questions are answered by Google Gemini, which receives the full event log
as context. No brittle regex rules — Gemini handles time ranges, counts,
durations, productivity judgements, and anything else naturally.
"""

from datetime import datetime


# ── Context builder ───────────────────────────────────────────────────────────

def _build_context(rows, limit=200) -> str:
    """Format DB rows into a readable event log for the LLM."""
    lines = []
    for ts_str, event in rows[:limit]:
        try:
            dt = datetime.fromisoformat(ts_str)
            ts_label = dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            ts_label = ts_str
        lines.append(f"[{ts_label}] {event}")
    return "\n".join(lines) if lines else "(no events recorded yet)"


# ── Gemini call ───────────────────────────────────────────────────────────────

def _gemini_answer(question: str, context_text: str, api_key: str) -> str:
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        system_prompt = (
            "You are a smart desk productivity assistant. "
            "You analyse a user's desk activity log and answer questions accurately.\n\n"

            "PRODUCTIVITY RUBRIC:\n"
            "• PRODUCTIVE: laptop present long stretches, person consistently at desk, "
            "minimal phone use, few/short absences.\n"
            "• UNPRODUCTIVE: frequent/long absences, many phone interactions (especially clusters), "
            "phone without laptop, very short desk presence.\n"
            "• MIXED: focused time interrupted by distractions.\n\n"

            "RULES:\n"
            "1. Read the event log carefully, paying attention to EXACT timestamps.\n"
            "2. When a question mentions a time (e.g. 'from 10:40', 'between 10:40 and 11:00'), "
            "ONLY consider events within that window — ignore everything outside it.\n"
            "3. For count questions ('how many times'), count only matching events in the time window.\n"
            "4. For productivity questions, give a clear verdict: PRODUCTIVE, UNPRODUCTIVE, or MIXED "
            "— never just summarise, always judge.\n"
            "5. Be concise and direct. Give the answer first, then 2-3 bullet-point observations.\n"
            "6. If the log has no data for the asked period, say so clearly.\n\n"

            "DESK EVENT LOG (most recent first):\n"
            f"{context_text}"
        )

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            ),
        )
        return response.text.strip()

    except ImportError:
        return "⚠️ google-genai is not installed. Run: pip install google-genai"
    except Exception as e:
        return f"⚠️ Gemini error: {e}"


# ── Main interface ────────────────────────────────────────────────────────────

def answer(question: str, db, api_key: str = "") -> dict:
    """
    Returns:
      {
        "answer":  str,         # natural-language answer
        "rows":    list[tuple], # supporting (timestamp, event) rows
        "used_ai": bool,
      }
    """
    all_rows = db.recent(500)

    if not api_key:
        return {
            "answer": (
                "💡 No Gemini API key set. "
                "Please enter your API key in the sidebar to enable AI answers."
            ),
            "rows": [],
            "used_ai": False,
        }

    context_text = _build_context(all_rows)
    ai_answer = _gemini_answer(question, context_text, api_key)

    return {
        "answer":  ai_answer,
        "rows":    all_rows[:20],
        "used_ai": True,
    }
