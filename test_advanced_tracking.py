from app.services.tracker import ObjectTracker
from app.services.analytics import TrackingAnalytics


VIDEO_PATH = "datasets/test.mp4"

tracker = ObjectTracker()
analytics = TrackingAnalytics()

results = tracker.track_video(VIDEO_PATH)

frame_number = 0
max_frames = 100

trajectories = {}


for result in results:

    frame_number += 1

    detections = tracker.get_tracked_objects(result)

    analytics.update(detections, frame_number)

    for detection in detections:

        track_id = detection["track_id"]

        if track_id is None:
            continue

        x1, y1, x2, y2 = detection["bbox"]

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        if track_id not in trajectories:
            trajectories[track_id] = []

        trajectories[track_id].append(
            (center_x, center_y)
        )

    if frame_number % 10 == 0:
        print(f"Processed {frame_number} frames...")

    if frame_number >= max_frames:
        break


print("\n========== TRACKING ANALYTICS ==========")

summary = analytics.get_summary()

for obj in summary:

    print(
        f"ID {obj['track_id']} | "
        f"Object: {obj['class']} | "
        f"Frames: {obj['frames_visible']} | "
        f"Avg Confidence: {obj['average_confidence']}"
    )


print("\n========== CLASS COUNTS ==========")

class_counts = analytics.get_class_counts()

for object_class, count in class_counts.items():

    print(
        f"{object_class}: {count}"
    )


print("\n========== TRAJECTORIES ==========")

for track_id, points in trajectories.items():

    print(
        f"ID {track_id} | "
        f"Trajectory points: {len(points)}"
    )