from ultralytics import YOLO
from pathlib import Path

from app.services.tracker import ObjectTracker


class ObjectDetector:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)
        self.tracker = ObjectTracker()

    def detect(self, image_path, confidence=0.25):
        results = self.model.predict(
            source=image_path,
            conf=confidence,
            verbose=False
        )

        return results[0]

    def get_detections(self, result, track=False):
        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class": self.model.names[class_id],
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })

        # Apply object tracking when requested
        if track:
            detections = self.tracker.update(detections)

        return detections

    def save_result(self, result, output_path):
        result.save(filename=str(output_path))

    def reset_tracker(self):
        self.tracker.reset()