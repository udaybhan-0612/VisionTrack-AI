from app.services.analytics import TrackingAnalytics


analytics = TrackingAnalytics()

frame_1 = [
    {
        "class": "person",
        "confidence": 0.90,
        "bbox": [100, 100, 200, 300],
        "track_id": 1
    },
    {
        "class": "car",
        "confidence": 0.85,
        "bbox": [400, 200, 600, 400],
        "track_id": 2
    }
]

frame_2 = [
    {
        "class": "person",
        "confidence": 0.92,
        "bbox": [105, 105, 205, 305],
        "track_id": 1
    },
    {
        "class": "car",
        "confidence": 0.87,
        "bbox": [405, 205, 605, 405],
        "track_id": 2
    }
]

analytics.update(frame_1, 1)
analytics.update(frame_2, 2)

print("\nTracking Summary:")
print(analytics.get_summary())

print("\nClass Counts:")
print(analytics.get_class_counts())
