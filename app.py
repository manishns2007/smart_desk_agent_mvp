import time
import threading
import cv2
import streamlit as st
import pandas as pd

from database import EventDB
from event_engine import EventEngine
from vision import Vision

st.set_page_config(page_title="Smart Desk Agent", layout="wide")
st.title("🧠 Smart Desk / Productivity Agent")
st.caption("Laptop camera → vision → event memory → natural-language queries")

# ── Singletons: survive across Streamlit reruns ─────────────────────────────
@st.cache_resource
def get_db():
    return EventDB()

@st.cache_resource
def get_vision():
    return Vision()

@st.cache_resource
def get_engine(db):
    return EventEngine(db)

db     = get_db()
vision = get_vision()
engine = get_engine(db)

# ── Session state ────────────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "latest_frame" not in st.session_state:
    st.session_state.latest_frame = None
if "latest_labels" not in st.session_state:
    st.session_state.latest_labels = []

# ── Background camera thread ─────────────────────────────────────────────────
_stop_event = threading.Event()

def _camera_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.session_state.running = False
        return
    try:
        while not _stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            labels, annotated = vision.process(frame)
            engine.update(labels)
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.session_state.latest_frame  = frame_rgb
            st.session_state.latest_labels = labels
            time.sleep(0.1)
    finally:
        cap.release()
        st.session_state.running = False

# ── Layout ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live camera")
    frame_box  = st.empty()
    status_box = st.empty()

    if st.session_state.latest_frame is not None:
        frame_box.image(
            st.session_state.latest_frame,
            channels="RGB",
            width='stretch',
        )
        labels = st.session_state.latest_labels
        status_box.info("Detected: " + (", ".join(labels) if labels else "nothing"))

with col2:
    st.subheader("Ask the agent")
    question = st.text_input("Question", placeholder="When did I last use my phone?")

    if question:
        q = question.lower()
        if "phone" in q:
            rows = db.search("phone", 10)
        elif "laptop" in q:
            rows = db.search("laptop", 10)
        elif "cup" in q or "drink" in q:
            rows = db.search("cup", 10)
        elif "book" in q or "write" in q:
            rows = db.search("book", 10)
        elif "enter" in q or "arrive" in q:
            rows = db.search("arrived", 10)
        elif "leave" in q or "away" in q:
            rows = db.search("left", 10)
        else:
            rows = db.recent(10)

        if rows:
            latest = rows[0]
            st.success(f"Latest relevant event: **{latest[1]}** at **{latest[0]}**")
            st.dataframe(
                pd.DataFrame(rows, columns=["Time", "Event"]),
                width='stretch',
                hide_index=True,
            )
        else:
            st.info("I don't have a matching event yet.")

# ── Event memory ─────────────────────────────────────────────────────────────
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

st.divider()

# ── Controls ─────────────────────────────────────────────────────────────────
start_btn = st.button("▶ Start camera", type="primary", disabled=st.session_state.running)
stop_btn  = st.button("⏹ Stop camera",  disabled=not st.session_state.running)

if start_btn and not st.session_state.running:
    _stop_event.clear()
    st.session_state.running = True
    t = threading.Thread(target=_camera_loop, daemon=True)
    t.start()
    st.rerun()

if stop_btn and st.session_state.running:
    _stop_event.set()
    st.session_state.running = False
    st.rerun()

if st.session_state.running:
    # Auto-refresh every second to show latest frame
    time.sleep(1)
    st.rerun()
else:
    if st.session_state.latest_frame is None:
        st.info("Click **Start camera** to begin observing the desk.")
