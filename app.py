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
        self.frame  = None   # numpy RGB array
        self.labels = []
        self.lock   = threading.Lock()
        self.running = False
        self.stop_event = threading.Event()

    def write(self, frame, labels):
        with self.lock:
            self.frame  = frame
            self.labels = labels

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
            time.sleep(0.05)
    finally:
        cap.release()
        frame_store.running = False

# ── Layout ───────────────────────────────────────────────────────────────────

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live camera")
    frame_box  = st.empty()
    status_box = st.empty()

    frame, labels = frame_store.read()
    if frame is not None:
        frame_box.image(frame, channels="RGB", width="stretch")
        status_box.info("Detected: " + (", ".join(labels) if labels else "nothing"))
    elif not frame_store.running:
        frame_box.info("Camera not started.")

with col2:
    st.subheader("Ask the agent")
    question = st.text_input(
        "Question",
        placeholder="How many times did I leave between 19:00 to 19:05?"
    )

    if question:
        result = agent_answer(question, db)
        st.success(result["answer"])

        if result["rows"]:
            st.dataframe(
                pd.DataFrame(result["rows"], columns=["Time", "Event"]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No supporting events to display.")

# ── Event memory ─────────────────────────────────────────────────────────────

st.subheader("Event memory")
rows = db.recent(30)
if rows:
    st.dataframe(
        pd.DataFrame(rows, columns=["Time", "Event"]),
        width="stretch",
        hide_index=True,
    )
else:
    st.info("No events recorded yet.")

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
    st.rerun()

# ── Auto-refresh while camera is live ────────────────────────────────────────

if frame_store.running:
    time.sleep(0.5)
    st.rerun()
