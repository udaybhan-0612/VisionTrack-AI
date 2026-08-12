from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Run object detection
results = model.predict(
    source="datasets/test.jpg",
    conf=0.25,
    save=True
)

print("\nDetection completed successfully!")

for result in results:
    print(f"Objects detected: {len(result.boxes)}")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            f"Object: {model.names[class_id]} "
            f"| Confidence: {confidence:.2f}"
        )
