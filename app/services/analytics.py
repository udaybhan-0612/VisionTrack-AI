from collections import defaultdict
import math


class TrackingAnalytics:

    def __init__(self):
        self.objects = defaultdict(lambda: {
            "class": None,
            "frames": 0,
            "confidence_sum": 0.0,
            "first_frame": None,
            "last_frame": None,
            "positions": [],
            "distance": 0.0
        })

    def update(self, detections, frame_number):

        for detection in detections:

            track_id = detection.get("track_id")

            if track_id is None:
                continue

            obj = self.objects[track_id]

            obj["class"] = detection["class"]
            obj["frames"] += 1
            obj["confidence_sum"] += detection["confidence"]

            if obj["first_frame"] is None:
                obj["first_frame"] = frame_number

            obj["last_frame"] = frame_number

            x1, y1, x2, y2 = detection["bbox"]

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            current_position = (center_x, center_y)

            if obj["positions"]:

                previous_x, previous_y = obj["positions"][-1]

                movement = math.sqrt(
                    (center_x - previous_x) ** 2 +
                    (center_y - previous_y) ** 2
                )

                obj["distance"] += movement

            obj["positions"].append(current_position)

    def get_summary(self):

        summary = []

        for track_id, obj in self.objects.items():

            average_confidence = (
                obj["confidence_sum"] / obj["frames"]
                if obj["frames"] > 0
                else 0
            )

            summary.append({
                "track_id": track_id,
                "class": obj["class"],
                "frames_visible": obj["frames"],
                "average_confidence": round(
                    average_confidence, 3
                ),
                "first_frame": obj["first_frame"],
                "last_frame": obj["last_frame"],
                "movement_distance": round(
                    obj["distance"], 2
                )
            })

        return summary

    def get_class_counts(self):

        counts = defaultdict(int)

        for obj in self.objects.values():
            counts[obj["class"]] += 1

        return dict(counts)