import numpy as np
import cv2
from ultralytics import YOLO

# COCO classes we care about — everything else is filtered from both
# the event log AND the on-screen bounding boxes.
INTERESTING = {
    "person",
    "cell phone",
    "laptop",
    "cup",
    "bottle",
    "book",
}

# Colour map for each interesting class (BGR)
_COLOURS = {
    "person":     (0,   200, 255),   # cyan
    "cell phone": (255, 100,   0),   # orange
    "laptop":     (0,   255, 120),   # green
    "cup":        (200,   0, 255),   # purple
    "bottle":     (255, 220,   0),   # yellow
    "book":       (0,   120, 255),   # blue
}

class Vision:
    def __init__(self, model_name="yolo11s.pt", confidence=0.45):
        """
        yolo11s (small) is used instead of yolo11n (nano) for better accuracy.
        Ultralytics auto-downloads the weights on first run.
        """
        self.model = YOLO(model_name)
        self.confidence = confidence

    def process(self, frame):
        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
        )[0]

        labels = []
        # Collect only interesting detections
        interesting_indices = []
        if result.boxes is not None:
            for i, cls_id in enumerate(result.boxes.cls.tolist()):
                label = self.model.names[int(cls_id)]
                if label in INTERESTING:
                    labels.append(label)
                    interesting_indices.append(i)

        # Draw ONLY interesting boxes on a clean copy of the frame
        annotated = frame.copy()
        if result.boxes is not None and interesting_indices:
            boxes  = result.boxes.xyxy.tolist()
            confs  = result.boxes.conf.tolist()
            cls_ids = result.boxes.cls.tolist()

            for i in interesting_indices:
                x1, y1, x2, y2 = [int(v) for v in boxes[i]]
                label = self.model.names[int(cls_ids[i])]
                conf  = confs[i]
                colour = _COLOURS.get(label, (255, 255, 255))

                # Box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

                # Label background + text
                text = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
                cv2.putText(
                    annotated, text,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 1, cv2.LINE_AA,
                )

        return sorted(set(labels)), annotated
