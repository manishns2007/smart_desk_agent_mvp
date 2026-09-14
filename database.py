import sqlite3
import threading
from datetime import datetime

class EventDB:
    def __init__(self, path="events.db"):
        self.path = path
        self.lock = threading.Lock()
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    epoch REAL NOT NULL,
                    event TEXT NOT NULL
                )
            """)

    def _connect(self):
        return sqlite3.connect(self.path, check_same_thread=False)

    def add_event(self, event, epoch):
        ts = datetime.fromtimestamp(epoch).isoformat(timespec="seconds")
        with self.lock, self._connect() as con:
            con.execute(
                "INSERT INTO events(timestamp, epoch, event) VALUES (?, ?, ?)",
                (ts, epoch, event),
            )

    def recent(self, limit=100):
        with self.lock, self._connect() as con:
            return con.execute(
                "SELECT timestamp, event FROM events ORDER BY epoch DESC LIMIT ?",
                (limit,),
            ).fetchall()

    def search(self, keyword, limit=20):
        with self.lock, self._connect() as con:
            return con.execute(
                "SELECT timestamp, event FROM events "
                "WHERE event LIKE ? ORDER BY epoch DESC LIMIT ?",
                (f"%{keyword}%", limit),
            ).fetchall()
