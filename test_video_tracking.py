from app.services.tracker import ObjectTracker
from app.services.analytics import TrackingAnalytics


tracker = ObjectTracker("yolo11n.pt")

analytics = TrackingAnalytics(fps=30)

video_path = "datasets/test.mp4"

results = tracker.track_video(
    video_path,
    confidence=0.25
)

frame_number = 0

for result in results:

    frame_number += 1

    objects = tracker.get_tracked_objects(result)

    analytics.update(objects)

    print(f"\nFrame {frame_number}")

    for obj in objects:

        print(
            f"ID: {obj['track_id']} | "
            f"Object: {obj['class']} | "
            f"Confidence: {obj['confidence']:.2f}"
        )

    if frame_number >= 100:
        break


print("\n" + "=" * 50)
print("TRACKING ANALYTICS")
print("=" * 50)

summary = analytics.get_summary()

print(f"Frames processed: {summary['frames_processed']}")
print(f"Unique objects: {summary['unique_objects']}")

print("\nClass detection counts:")

for cls, count in summary["class_counts"].items():
    print(f"{cls}: {count}")

print("\nAverage confidence:")

for cls, confidence in summary["average_confidence"].items():
    print(f"{cls}: {confidence:.2f}")

print("\nTracked objects:")

for track_id, data in summary["tracked_objects"].items():

    print(
        f"ID {track_id} | "
        f"{data['class']} | "
        f"Visible for {data['frames_visible']} frames"
    )
