"""
query_engine.py – Natural-language query interpreter for Smart Desk Agent.

Understands (fast regex paths – no API call):
  - Time ranges  : "between 19:00 to 19:05", "from 18:50 to 19:00"
  - Counts        : "how many times", "how often"
  - Last-seen     : "when did i last", "last time"
  - Duration      : "how long was i", "how long did i"
  - Topic keywords: person / phone / laptop / cup / book / left / arrived

Freeform / complex questions fall through to Google Gemini Flash.
"""

import re
from datetime import datetime, date

# ── Gemini client (lazy-initialised per call) ─────────────────────────────────

def _gemini_answer(question: str, context_text: str, api_key: str) -> str:
    """Send question + event context to Gemini and return the answer string."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        system_prompt = (
            "You are a smart desk productivity assistant that analyses a user's desk activity log "
            "and gives clear, opinionated answers.\n\n"
            "PRODUCTIVITY RUBRIC (use this to judge sessions):\n"
            "• PRODUCTIVE signals: laptop present for long stretches, person consistently at desk, "
            "minimal phone interactions, few/short desk absences.\n"
            "• UNPRODUCTIVE signals: frequent or long desk absences, many phone interactions "
            "(especially in clusters), phone detected without laptop, very short bursts of desk presence.\n"
            "• MIXED: some focused time interrupted by distractions.\n\n"
            "RULES:\n"
            "1. Always give a clear verdict: PRODUCTIVE, UNPRODUCTIVE, or MIXED — do NOT just summarise.\n"
            "2. Back the verdict with 2-3 specific observations from the log (times + events).\n"
            "3. Be concise and direct. One verdict line, then bullet-point evidence.\n"
            "4. If the log has too little data, say so and give a best-guess verdict anyway.\n\n"
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


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_context(rows, limit=150) -> str:
    """Format DB rows into a compact event log string for the LLM."""
    lines = []
    for ts_str, event in rows[:limit]:
        # Shorten ISO timestamp → HH:MM:SS for readability
        try:
            dt = datetime.fromisoformat(ts_str)
            ts_label = dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            ts_label = ts_str
        lines.append(f"[{ts_label}] {event}")
    return "\n".join(lines) if lines else "(no events recorded yet)"


def _topic_keyword(q: str) -> str | None:
    """Return the DB search keyword that best matches the question."""
    mapping = [
        (["phone", "cell", "mobile"],        "phone"),
        (["laptop", "computer", "pc"],        "laptop"),
        (["cup", "drink", "coffee", "water"], "cup"),
        (["book", "write", "reading"],        "book"),
        (["left", "leave", "away", "gone"],   "left"),
        (["arrived", "enter", "came", "back", "return"], "arrived"),
        (["person", "desk", "sitting"],       "person"),
    ]
    for triggers, keyword in mapping:
        if any(t in q for t in triggers):
            return keyword
    return None


def _parse_time(text: str) -> datetime | None:
    """Parse loose time strings like '19:00', '19 : 00', '7pm' into a datetime today."""
    # normalise spaces around colons
    text = re.sub(r"\s*:\s*", ":", text.strip())
    today = date.today()
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M%p", "%I%p"):
        try:
            t = datetime.strptime(text.upper(), fmt).time()
            return datetime.combine(today, t)
        except ValueError:
            pass
    return None


def _extract_time_range(q: str):
    """Return (start_dt, end_dt) or (None, None) if no range found."""
    # patterns: "between X to Y", "between X and Y", "from X to Y", "from X - Y"
    pattern = r"(?:between|from)\s+(.+?)\s+(?:to|and|-)\s+(.+?)(?:\s|$|,|\?)"
    m = re.search(pattern, q)
    if m:
        t1 = _parse_time(m.group(1))
        t2 = _parse_time(m.group(2))
        if t1 and t2:
            return t1, t2
    return None, None


def _filter_by_range(rows, start_dt, end_dt):
    """rows: list of (timestamp_str, event_str). Returns filtered list."""
    result = []
    for ts_str, event in rows:
        try:
            ts = datetime.fromisoformat(ts_str)
            if start_dt <= ts <= end_dt:
                result.append((ts_str, event))
        except ValueError:
            pass
    return result


def _duration_between(rows, start_event_kw, end_event_kw):
    """Calculate total duration between paired start/end events (minutes)."""
    starts = []
    periods = []
    for ts_str, event in sorted(rows, key=lambda r: r[0]):
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            continue
        if start_event_kw in event.lower():
            starts.append(ts)
        elif end_event_kw in event.lower() and starts:
            s = starts.pop()
            periods.append((ts - s).total_seconds() / 60)
    return periods


# ── Main interface ────────────────────────────────────────────────────────────

def answer(question: str, db, api_key: str = "") -> dict:
    """
    Returns a dict:
      {
        "answer": str,          # the direct natural-language answer
        "rows":   list[tuple],  # (timestamp, event) rows to display (may be [])
        "count":  int | None,   # if a count was computed
        "used_ai": bool,        # True if Gemini was called
      }
    """
    q = question.lower()
    all_recent = db.recent(500)   # pull enough history to work with

    # Determine topic
    keyword = _topic_keyword(q)
    if keyword:
        topic_rows = [r for r in all_recent if keyword in r[1].lower()]
    else:
        topic_rows = all_recent

    # Detect time range
    start_dt, end_dt = _extract_time_range(q)
    time_range_str = ""
    if start_dt and end_dt:
        topic_rows = _filter_by_range(topic_rows, start_dt, end_dt)
        time_range_str = f" between {start_dt.strftime('%H:%M')} and {end_dt.strftime('%H:%M')}"

    # ── Intent: COUNT ────────────────────────────────────────────────────────
    if any(w in q for w in ["how many", "how often", "count", "number of times"]):
        count = len(topic_rows)
        subject = keyword or "event"
        if count == 0:
            answer_text = f"No '{subject}' events found{time_range_str}."
        elif count == 1:
            answer_text = f"**1 time** — {subject}{time_range_str}."
        else:
            answer_text = f"**{count} times** — {subject}{time_range_str}."
        return {"answer": answer_text, "rows": topic_rows, "count": count, "used_ai": False}

    # ── Intent: LAST SEEN ────────────────────────────────────────────────────
    if any(w in q for w in ["when did", "last time", "last seen", "when was", "when were"]):
        if topic_rows:
            ts, event = topic_rows[0]   # most recent first
            answer_text = f"Last event: **{event}** at **{ts}**"
        else:
            answer_text = f"No matching events found{time_range_str}."
        return {"answer": answer_text, "rows": topic_rows[:10], "count": None, "used_ai": False}

    # ── Intent: DURATION ─────────────────────────────────────────────────────
    if any(w in q for w in ["how long", "duration", "total time"]):
        periods = _duration_between(all_recent, "arrived", "ended")
        if start_dt and end_dt:
            periods = _duration_between(topic_rows, "arrived", "ended")
        if periods:
            total = sum(periods)
            sessions = len(periods)
            answer_text = (
                f"**{total:.1f} min** across {sessions} session(s){time_range_str}. "
                f"Avg: {total/sessions:.1f} min/session."
            )
        else:
            answer_text = f"Not enough data to calculate duration{time_range_str}."
        return {"answer": answer_text, "rows": topic_rows[:20], "count": None, "used_ai": False}

    # ── Fallback: Gemini AI ───────────────────────────────────────────────────
    context_text = _build_context(all_recent)

    if not api_key:
        # No key — give best-effort plain answer from the data
        display = topic_rows[:20]
        if display:
            ts, event = display[0]
            answer_text = f"Latest: **{event}** at **{ts}**{time_range_str}"
        else:
            answer_text = (
                f"No matching events found{time_range_str}. "
                "💡 Add a Gemini API key in the sidebar to enable AI answers for complex questions."
            )
        return {"answer": answer_text, "rows": display, "count": None, "used_ai": False}

    ai_answer = _gemini_answer(question, context_text, api_key)
    return {"answer": ai_answer, "rows": topic_rows[:20], "count": None, "used_ai": True}
