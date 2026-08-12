from ultralytics import YOLO


class ObjectTracker:

    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)

    def track_video(self, video_path, confidence=0.25):
        results = self.model.track(
            source=video_path,
            conf=confidence,
            persist=True,
            tracker="bytetrack.yaml",
            stream=True,
            verbose=False
        )

        return results

    def get_tracked_objects(self, result):
        tracked_objects = []

        if result.boxes is None:
            return tracked_objects

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            track_id = None

            if box.id is not None:
                track_id = int(box.id[0])

            tracked_objects.append({
                "class": self.model.names[class_id],
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
                "track_id": track_id
            })

        return tracked_objects