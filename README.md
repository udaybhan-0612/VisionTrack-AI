# 🎯 VisionTrack AI
### Internship Project — Object Detection & Multi-Object Tracking

| Internship Detail | Information |
|-------------------|-------------|
| **Intern Name** | Udaybhan Pandey |
| **Intern ID** | CITS8162 |
| **Project Name** | VisionTrack AI |
| **Project Scope** | Intelligent Object Detection, Multi-Object Tracking & Computer Vision Analytics |
| **Duration** | 1 Week |
| **Project Domain** | Artificial Intelligence & Computer Vision |


### Intelligent Object Detection, Multi-Object Tracking & Computer Vision Analytics

VisionTrack AI is an end-to-end computer vision application built using **YOLO11, ByteTrack, OpenCV, Python, and Streamlit**.

The system can analyze images, videos, and camera captures to detect objects, track them across video frames, visualize movement trajectories, calculate confidence and performance metrics, and generate downloadable analytical reports.

---

## 🚀 Project Overview

Traditional object detection systems identify objects independently in each frame.

VisionTrack AI goes one step further by combining:

- Object Detection
- Multi-Object Tracking
- Persistent Track IDs
- Object Trajectories
- Confidence Analysis
- Performance Monitoring
- Statistical Analytics
- Automated Report Generation

This transforms a basic object detection model into an interactive computer vision analytics platform.

---

## ✨ Key Features

### 🔍 Object Detection

- YOLO11-based object detection
- Configurable confidence threshold
- Bounding box visualization
- Object class identification
- Confidence score visualization

### 🎥 Multi-Object Tracking

- ByteTrack-based object tracking
- Persistent tracking IDs
- Object tracking across multiple frames
- Frame-level detection analysis
- Movement trajectory visualization

### 📊 Computer Vision Analytics

The application provides:

- Total objects detected
- Unique object classes
- Average confidence
- Processing FPS
- Processing time
- Video FPS
- Video duration
- Frames processed
- Objects tracked
- Object class distribution

### 🛤️ Trajectory Analysis

Tracked objects are represented with movement trajectories.

The system records the center position of tracked objects across frames and visualizes their movement paths.

This helps analyze object movement throughout a video.

### 📥 Report Generation

Users can download analytical results in:

- CSV format
- JSON format

Reports include information such as:

- Track ID
- Object class
- Frames visible
- Average confidence
- First detected frame
- Last detected frame
- Movement distance

### 📡 Camera Detection

VisionTrack AI also supports camera-based object detection.

Users can capture an image using their device camera and run YOLO inference directly through the application.

---

# 🧠 System Architecture

```text
                        ┌─────────────────────┐
                        │    VisionTrack AI   │
                        │    Streamlit UI     │
                        └──────────┬──────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                │                  │                  │
                ▼                  ▼                  ▼
        ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
        │ Image Input  │   │ Video Input  │   │ Camera Input │
        └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
               │                  │                  │
               └──────────────────┼──────────────────┘
                                  ▼
                         ┌─────────────────┐
                         │     YOLO11      │
                         │ Object Detection│
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    ByteTrack    │
                         │ Object Tracking │
                         └────────┬────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
              ┌──────────┐ ┌───────────┐ ┌────────────┐
              │Tracking  │ │Trajectory │ │Confidence  │
              │  IDs     │ │ Analysis  │ │ Analytics  │
              └────┬─────┘ └─────┬─────┘ └─────┬──────┘
                   │             │             │
                   └─────────────┼─────────────┘
                                 ▼
                       ┌────────────────────┐
                       │ Analytics Dashboard│
                       └──────────┬─────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
                 Charts        Reports       Metrics
                                  │
                           ┌──────┴──────┐
                           ▼             ▼
                         CSV           JSON