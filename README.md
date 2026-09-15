# 🧠 Smart Desk Agent

> A real-time AI-powered desk monitoring assistant — your laptop camera watches your desk, logs what happens, and lets you ask anything about your day in plain English.

---

## ✨ Features

- 📷 **Real-time object detection** — detects person, phone, laptop, cup, bottle, book using YOLO11
- 🗂️ **Automatic event logging** — timestamps every arrival, departure, and object appearance into a local SQLite database
- 🤖 **AI-powered Q&A** — ask anything about your desk activity in natural language, powered by Google Gemini
- 📊 **Live event memory** — scrollable table of everything that happened at your desk
- 🔒 **Fully local** — your camera feed never leaves your machine; only your questions go to Gemini

---

## 🖥️ Demo

| Live Camera | Ask the Agent |
|---|---|
| YOLO detects objects in real-time | *"How many times did I leave between 10:40 and 11:00?"* |
| Only relevant objects are highlighted | *"Was I productive this morning?"* |
| Events logged with exact timestamps | *"When did I last use my phone?"* |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** — [download here](https://www.python.org/downloads/)
- **Webcam** — built-in laptop camera works perfectly
- **Gemini API key** — free at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

---

### 1. Clone the repository

```bash
git clone https://github.com/your-username/smart_desk_agent_mvp.git
cd smart_desk_agent_mvp
```

---

### 2. Create a virtual environment (recommended)

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run, the YOLO model weights (~18 MB) will be downloaded automatically. This takes about 30 seconds depending on your connection.

---

### 4. Run the app

**Windows (double-click):**
```
run.bat
```

**Or from terminal:**
```bash
python -m streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

### 5. Enter your Gemini API key

1. Open the **sidebar** (click `>` on the top left)
2. Paste your Gemini API key (starts with `AQ...`)
3. You're ready to ask questions!

---

## 🗂️ Project Structure

```
smart_desk_agent_mvp/
│
├── app.py              # Streamlit UI — camera feed, Q&A panel, event table
├── vision.py           # YOLO11 wrapper — detects and filters objects
├── event_engine.py     # Converts detections → stable timestamped events
├── database.py         # SQLite event store
├── query_engine.py     # Gemini AI query interface
│
├── requirements.txt    # Python dependencies
├── run.bat             # One-click Windows launcher
└── .gitignore          # Excludes events.db, model weights, cache
```

---

## 💬 Example Questions

Once the camera is running and events are logged, try asking:

| Question | What it does |
|---|---|
| *How many times did I leave my desk today?* | Counts departure events |
| *When did I last use my phone?* | Finds most recent phone event |
| *Was I productive between 10am and 12pm?* | AI judges based on activity patterns |
| *How long was I at my desk this morning?* | Calculates total presence time |
| *What was I doing around 3pm?* | Summarises activity near that time |

---

## ⚙️ How It Works

```
Webcam frame (every 0.1s)
        │
        ▼
   YOLO11s model
   (local, offline)
        │
        ▼
  Filter: person / phone /        ←── irrelevant objects (cupboard,
  laptop / cup / bottle / book         door etc.) are hidden
        │
        ▼
  EventEngine — debounces detections
  into stable start/end events
        │
        ▼
  SQLite (events.db) — local,
  timestamped event log
        │
        ▼
  Your question → Gemini AI
  (event log sent as context)
        │
        ▼
  Natural-language answer
```

---

## 🔧 Configuration

| Setting | Default | How to change |
|---|---|---|
| Detection confidence | 0.45 | Edit `confidence=` in `vision.py` |
| YOLO model | `yolo11s.pt` | Change `model_name=` in `vision.py` (n/s/m/l/x) |
| Camera source | `0` (default webcam) | Change `VideoCapture(0)` in `app.py` |
| Event persistence | 2s | Edit `persistence_seconds` in `event_engine.py` |
| Disappearance timeout | 3s | Edit `disappearance_seconds` in `event_engine.py` |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `ultralytics` | YOLO11 object detection |
| `opencv-python` | Camera capture |
| `google-genai` | Gemini AI Q&A |
| `pandas` | Event table display |
| `Pillow` | Image handling |

---

## ⚠️ Limitations

- **Webcam required** — this app cannot be deployed to cloud servers (no camera available there)
- **Phone detection is proximity-based** — phone near person is flagged as interaction; true hand tracking not yet implemented
- **COCO class constraints** — YOLO is trained on COCO classes; room-specific objects (cupboard, door) won't be recognised correctly without a custom model
- **Privacy** — all video stays local; only text (your questions + event descriptions) is sent to Gemini

---

## 🛠️ Troubleshooting

**`streamlit` not recognised in terminal**
```bash
# Use the full Python path:
python -m streamlit run app.py
```

**Camera not opening**
- Make sure no other app (Teams, Zoom, etc.) is using the webcam
- Try changing `VideoCapture(0)` to `VideoCapture(1)` in `app.py` if you have multiple cameras

**Gemini error / no answer**
- Check your API key is correct (starts with `AQ...`)
- Get a free key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

**YOLO model download fails**
- Check your internet connection — Ultralytics downloads from GitHub on first run
- The model is saved as `yolo11s.pt` in the project folder after download

---

## 📄 License

MIT
