from ultralytics import YOLO

# COCO classes used by the prototype.
INTERESTING = {
    "person",
    "cell phone",
    "laptop",
    "cup",
    "bottle",
    "book",
}

class Vision:
    def __init__(self, model_name="yolo11n.pt", confidence=0.45):
        self.model = YOLO(model_name)
        self.confidence = confidence

    def process(self, frame):
        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
        )[0]

        labels = []
        if result.boxes is not None:
            for cls_id in result.boxes.cls.tolist():
                label = self.model.names[int(cls_id)]
                if label in INTERESTING:
                    labels.append(label)

        annotated = result.plot()
        return sorted(set(labels)), annotated
