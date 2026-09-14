import time
from collections import defaultdict

class EventEngine:
    """Converts frame-level detections into stable events using persistence."""

    def __init__(self, db, persistence_seconds=2.0, disappearance_seconds=3.0):
        self.db = db
        self.persistence = persistence_seconds
        self.disappearance = disappearance_seconds
        self.first_seen = {}
        self.last_seen = {}
        self.active = set()

    def update(self, labels):
        now = time.time()
        labels = set(labels)

        # Start events only after an object is continuously visible for a short period.
        for label in labels:
            self.last_seen[label] = now
            if label not in self.first_seen:
                self.first_seen[label] = now

            if label not in self.active and now - self.first_seen[label] >= self.persistence:
                self.active.add(label)
                self.db.add_event(self._start_event(label), now)

        # End events after disappearance.
        for label in list(self.active):
            if label not in labels:
                last = self.last_seen.get(label, now)
                if now - last >= self.disappearance:
                    self.active.remove(label)
                    self.first_seen.pop(label, None)
                    self.db.add_event(self._end_event(label), now)

        # Derived event: phone interaction is approximated.
        if "person" in labels and "cell phone" in labels:
            key = "phone_interaction"
            if key not in self.active:
                if "person" in self.active and "cell phone" in self.active:
                    self.active.add(key)
                    self.db.add_event("phone interaction detected", now)
        else:
            self.active.discard("phone_interaction")

    @staticmethod
    def _start_event(label):
        names = {
            "person": "person arrived / desk activity started",
            "cell phone": "phone detected",
            "laptop": "laptop detected",
            "cup": "cup detected",
            "bottle": "bottle detected",
            "book": "book detected",
        }
        return names.get(label, f"{label} detected")

    @staticmethod
    def _end_event(label):
        names = {
            "person": "person left / desk activity ended",
            "cell phone": "phone no longer visible",
            "laptop": "laptop no longer visible",
            "cup": "cup no longer visible",
            "bottle": "bottle no longer visible",
            "book": "book no longer visible",
        }
        return names.get(label, f"{label} no longer visible")
