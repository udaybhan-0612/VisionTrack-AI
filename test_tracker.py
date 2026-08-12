from app.services.tracker import ObjectTracker

tracker = ObjectTracker()

detections_frame_1 = [
    {
        "class": "person",
        "confidence": 0.90,
        "bbox": [100, 100, 200, 300]
    },
    {
        "class": "car",
        "confidence": 0.85,
        "bbox": [400, 200, 600, 400]
    }
]

detections_frame_2 = [
    {
        "class": "person",
        "confidence": 0.91,
        "bbox": [105, 105, 205, 305]
    },
    {
        "class": "car",
        "confidence": 0.86,
        "bbox": [405, 205, 605, 405]
    }
]

result1 = tracker.update(detections_frame_1)
result2 = tracker.update(detections_frame_2)

print("Frame 1:")
print(result1)

print("\nFrame 2:")
print(result2)
