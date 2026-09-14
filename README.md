# Smart Desk / Productivity Agent

Laptop-camera prototype that turns visual observations into timestamped events and lets the user ask questions about what happened.

## MVP detections
- person at/away from desk (approximated by person presence in camera view)
- cell phone detected
- laptop detected
- cup/bottle detected
- book detected
- object appeared/disappeared
- person entered/left

The system stores events in SQLite and exposes a Streamlit dashboard.

## Run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the YOLO model automatically.

## Notes
This is an MVP. "Phone used" is inferred from a phone being detected while a person is present; it is not yet true hand-object interaction tracking. The next upgrade is pose/hand tracking + object tracking.
