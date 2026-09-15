import time
import threading
import cv2
import streamlit as st
import pandas as pd

from database import EventDB
from event_engine import EventEngine
from vision import Vision
from query_engine import answer as agent_answer

st.set_page_config(page_title="Smart Desk Agent", layout="wide")
st.title("🧠 Smart Desk / Productivity Agent")
st.caption("Laptop camera → vision → event memory → natural-language queries")

# ── Sidebar: Gemini API Key ───────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AQ...",
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )
    if api_key:
        st.success("✅ Gemini AI enabled — ask anything!")
    else:
        st.info(
            "💡 **No API key set.**\n\n"
            "Structured queries (counts, last-seen, durations) work without a key.\n"
            "For freeform AI answers, enter your Gemini API key above."
        )
    st.divider()
    st.markdown(
        "**Example questions**\n"
        "- How many times did I leave today?\n"
        "- When did I last use my phone?\n"
        "- Was I productive this morning?\n"
        "- What was I doing around 10am?\n"
        "- How long was I at my desk?"
    )

# ── Singletons: survive across Streamlit reruns ──────────────────────────────

@st.cache_resource
def get_db():
    return EventDB()

@st.cache_resource
def get_vision():
    return Vision()

@st.cache_resource
def get_engine(_db):
    return EventEngine(_db)

class FrameStore:
    """Thread-safe container for the latest camera frame."""
    def __init__(self):
        self.frame   = None   # numpy RGB array, or None when stopped
        self.labels  = []
        self.lock    = threading.Lock()
        self.running = False
        self.stop_event = threading.Event()

    def write(self, frame, labels):
        with self.lock:
            self.frame  = frame
            self.labels = labels

    def clear(self):
        """Erase the last frame so the feed goes blank on stop."""
        with self.lock:
            self.frame  = None
            self.labels = []

    def read(self):
        with self.lock:
            return self.frame, list(self.labels)

@st.cache_resource
def get_frame_store():
    return FrameStore()

db          = get_db()
vision      = get_vision()
engine      = get_engine(db)
frame_store = get_frame_store()

# ── Background camera thread ─────────────────────────────────────────────────

def _camera_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        frame_store.running = False
        return
    try:
        while not frame_store.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            labels, annotated = vision.process(frame)
            engine.update(labels)
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_store.write(frame_rgb, labels)
            time.sleep(0.1)   # ~10 fps — enough for desk monitoring, lighter on CPU
    finally:
        cap.release()
        frame_store.clear()   # blank the feed when thread exits
        frame_store.running = False

# ── Layout ───────────────────────────────────────────────────────────────────

col1, col2 = st.columns([2, 1])

# Camera column — refreshes independently via @st.fragment so the Q&A panel
# and event table don't rerender on every tick.
with col1:
    st.subheader("Live camera")

    @st.fragment(run_every=1)   # refresh this block every 1 s while camera is on
    def _camera_panel():
        frame, labels = frame_store.read()
        if frame_store.running and frame is not None:
            st.image(frame, channels="RGB", width='stretch')
            st.info("Detected: " + (", ".join(labels) if labels else "nothing"))
        elif frame_store.running:
            st.info("⏳ Starting camera…")
        else:
            st.info("📷 Camera stopped. Press **▶ Start camera** below.")

    _camera_panel()

with col2:
    st.subheader("Ask the agent")
    question = st.text_input(
        "Question",
        placeholder="How many times did I leave between 19:00 to 19:05?"
    )

    if question:
        with st.spinner("Thinking…"):
            result = agent_answer(question, db, api_key=api_key)

        if result.get("used_ai"):
            st.caption("🤖 Answered by Gemini AI")
        else:
            st.caption("⚡ Answered by fast rule engine")

        st.success(result["answer"])

        if result["rows"]:
            st.dataframe(
                pd.DataFrame(result["rows"], columns=["Time", "Event"]),
                width='stretch',
                hide_index=True,
            )
        else:
            st.info("No supporting events to display.")

# ── Event memory ─────────────────────────────────────────────────────────────

@st.fragment(run_every=5)
def _event_memory_panel():
    st.subheader("Event memory")
    rows = db.recent(30)
    if rows:
        st.dataframe(
            pd.DataFrame(rows, columns=["Time", "Event"]),
            width='stretch',
            hide_index=True,
        )
    else:
        st.info("No events recorded yet.")

_event_memory_panel()


st.divider()

# ── Controls ─────────────────────────────────────────────────────────────────

start_btn = st.button("▶ Start camera", type="primary", disabled=frame_store.running)
stop_btn  = st.button("⏹ Stop camera",  disabled=not frame_store.running)

if start_btn and not frame_store.running:
    frame_store.stop_event.clear()
    frame_store.running = True
    t = threading.Thread(target=_camera_loop, daemon=True)
    t.start()
    st.rerun()

if stop_btn and frame_store.running:
    frame_store.stop_event.set()
    frame_store.running = False
    frame_store.clear()   # immediately blank the feed, don't wait for thread
    st.rerun()
