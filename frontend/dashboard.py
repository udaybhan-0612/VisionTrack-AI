import streamlit as st
from pathlib import Path
import tempfile
import sys
import os
import cv2
import time
import json
import pandas as pd


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


from app.services.detector import ObjectDetector
from app.services.tracker import ObjectTracker
from app.services.analytics import TrackingAnalytics


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="VisionTrack AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# HEADER
# =========================================================

st.title("🎯 VisionTrack AI")

st.markdown(
    """
    ### Intelligent Object Detection & Multi-Object Tracking

    VisionTrack AI combines **YOLO11 object detection**,
    **ByteTrack multi-object tracking**, trajectory analysis,
    and performance analytics to analyze images and videos.
    """
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

st.sidebar.info(
    "Higher confidence thresholds produce fewer but more reliable detections."
)

st.sidebar.divider()

max_frames = st.sidebar.number_input(
    "Maximum Video Frames",
    min_value=50,
    max_value=2000,
    value=300,
    step=50
)

st.sidebar.caption(
    "Lower values are faster on CPU-based systems."
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    ### 🧠 VisionTrack AI

    **Detection:** YOLO11  
    **Tracking:** ByteTrack  
    **Analytics:** Custom Tracking Engine  
    **Interface:** Streamlit
    """
)


# =========================================================
# MODE SELECTION
# =========================================================

mode = st.radio(
    "Select Analysis Mode",
    [
        "📷 Image Detection",
        "🎥 Video Tracking",
        "📡 Live Camera"
    ],
    horizontal=True
)


# =========================================================
# IMAGE DETECTION
# =========================================================

if mode == "📷 Image Detection":

    st.subheader("📷 Image Detection")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        key="image_upload"
    )

    if uploaded_file:

        col1, col2 = st.columns(2)

        # -------------------------------------------------
        # SAVE TEMPORARY IMAGE
        # -------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(uploaded_file.name).suffix
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            input_path = temp_file.name

        # -------------------------------------------------
        # ORIGINAL IMAGE
        # -------------------------------------------------

        with col1:

            st.markdown(
                "### 🖼️ Original Image"
            )

            st.image(
                uploaded_file,
                use_container_width=True
            )

        # -------------------------------------------------
        # DETECTION
        # -------------------------------------------------

        if st.button(
            "🚀 Detect Objects",
            type="primary",
            use_container_width=True
        ):

            start_time = time.time()

            with st.spinner(
                "Running YOLO detection..."
            ):

                detector = ObjectDetector()

                result = detector.detect(
                    input_path,
                    confidence
                )

                detections = (
                    detector.get_detections(
                        result
                    )
                )

            processing_time = (
                time.time() - start_time
            )

            # -------------------------------------------------
            # RESULT IMAGE
            # -------------------------------------------------

            with col2:

                st.markdown(
                    "### 🎯 Detection Result"
                )

                annotated_image = result.plot()

                st.image(
                    annotated_image,
                    channels="BGR",
                    use_container_width=True
                )

            st.divider()

            # -------------------------------------------------
            # ANALYTICS
            # -------------------------------------------------

            st.subheader(
                "📊 Detection Analytics"
            )

            total_objects = len(
                detections
            )

            class_counts = {}

            for detection in detections:

                class_name = detection[
                    "class"
                ]

                class_counts[class_name] = (
                    class_counts.get(
                        class_name,
                        0
                    ) + 1
                )

            if detections:

                avg_confidence = (
                    sum(
                        d["confidence"]
                        for d in detections
                    )
                    / len(detections)
                )

            else:

                avg_confidence = 0

            # -------------------------------------------------
            # METRICS
            # -------------------------------------------------

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            with col1:

                st.metric(
                    "Objects Detected",
                    total_objects
                )

            with col2:

                st.metric(
                    "Unique Classes",
                    len(class_counts)
                )

            with col3:

                st.metric(
                    "Average Confidence",
                    f"{avg_confidence * 100:.2f}%"
                )

            with col4:

                st.metric(
                    "Processing Time",
                    f"{processing_time:.2f}s"
                )

            # -------------------------------------------------
            # CLASS DISTRIBUTION
            # -------------------------------------------------

            if class_counts:

                st.markdown(
                    "### 📈 Object Class Distribution"
                )

                chart_data = pd.DataFrame(
                    {
                        "Objects": class_counts
                    }
                )

                st.bar_chart(
                    chart_data
                )

            # -------------------------------------------------
            # OBJECT BREAKDOWN
            # -------------------------------------------------

            st.markdown(
                "### 🔎 Object Breakdown"
            )

            for class_name, count in (
                class_counts.items()
            ):

                class_detections = [
                    d
                    for d in detections
                    if d["class"] == class_name
                ]

                avg_conf = (
                    sum(
                        d["confidence"]
                        for d in class_detections
                    )
                    / len(class_detections)
                )

                st.write(
                    f"**{class_name.title()}** — "
                    f"{count} detected — "
                    f"{avg_conf * 100:.1f}% "
                    f"average confidence"
                )

            # -------------------------------------------------
            # IMAGE REPORT
            # -------------------------------------------------

            if detections:

                image_report = []

                for detection in detections:

                    image_report.append(
                        {
                            "Object":
                                detection["class"],

                            "Confidence":
                                round(
                                    detection[
                                        "confidence"
                                    ],
                                    4
                                ),

                            "X1":
                                round(
                                    detection[
                                        "bbox"
                                    ][0],
                                    2
                                ),

                            "Y1":
                                round(
                                    detection[
                                        "bbox"
                                    ][1],
                                    2
                                ),

                            "X2":
                                round(
                                    detection[
                                        "bbox"
                                    ][2],
                                    2
                                ),

                            "Y2":
                                round(
                                    detection[
                                        "bbox"
                                    ][3],
                                    2
                                )
                        }
                    )

                report_df = pd.DataFrame(
                    image_report
                )

                st.markdown(
                    "### 📥 Detection Report"
                )

                st.download_button(
                    "⬇️ Download CSV Report",
                    report_df.to_csv(
                        index=False
                    ),
                    file_name=(
                        "visiontrack_image_report.csv"
                    ),
                    mime="text/csv"
                )

    else:

        st.info(
            "👆 Upload an image above to begin object detection."
        )


# =========================================================
# VIDEO TRACKING
# =========================================================

elif mode == "🎥 Video Tracking":

    st.subheader(
        "🎥 Multi-Object Video Tracking"
    )

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ],
        key="video_upload"
    )

    if uploaded_video:

        st.video(
            uploaded_video
        )

        if st.button(
            "🚀 Start Video Tracking",
            type="primary",
            use_container_width=True
        ):

            start_time = time.time()

            # -------------------------------------------------
            # SAVE INPUT VIDEO
            # -------------------------------------------------

            video_suffix = Path(
                uploaded_video.name
            ).suffix

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=video_suffix
            ) as temp_input:

                temp_input.write(
                    uploaded_video.getbuffer()
                )

                input_video_path = (
                    temp_input.name
                )

            # -------------------------------------------------
            # VIDEO INFORMATION
            # -------------------------------------------------

            cap = cv2.VideoCapture(
                input_video_path
            )

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            total_video_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            cap.release()

            if fps <= 0:

                fps = 25

            video_duration = (
                total_video_frames / fps
                if fps > 0
                else 0
            )

            # -------------------------------------------------
            # OUTPUT VIDEO
            # -------------------------------------------------

            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_video_path = (
                output_file.name
            )

            output_file.close()

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                output_video_path,
                fourcc,
                fps,
                (width, height)
            )

            # -------------------------------------------------
            # TRACKING ENGINE
            # -------------------------------------------------

            tracker = ObjectTracker()

            analytics = TrackingAnalytics()

            trajectories = {}

            frame_number = 0

            progress = st.progress(0)

            status = st.empty()

            # -------------------------------------------------
            # TRACK VIDEO
            # -------------------------------------------------

            results = tracker.track_video(
                input_video_path,
                confidence
            )

            for result in results:

                frame_number += 1

                detections = (
                    tracker.get_tracked_objects(
                        result
                    )
                )

                analytics.update(
                    detections,
                    frame_number
                )

                # -------------------------------------------------
                # STORE TRAJECTORY
                # -------------------------------------------------

                for detection in detections:

                    track_id = (
                        detection.get(
                            "track_id"
                        )
                    )

                    if track_id is None:

                        continue

                    x1, y1, x2, y2 = (
                        detection["bbox"]
                    )

                    center_x = int(
                        (x1 + x2) / 2
                    )

                    center_y = int(
                        (y1 + y2) / 2
                    )

                    if track_id not in trajectories:

                        trajectories[
                            track_id
                        ] = []

                    trajectories[
                        track_id
                    ].append(
                        (
                            center_x,
                            center_y
                        )
                    )

                # -------------------------------------------------
                # DRAW YOLO RESULT
                # -------------------------------------------------

                frame = result.plot()

                # -------------------------------------------------
                # DRAW TRAJECTORIES
                # -------------------------------------------------

                for (
                    track_id,
                    points
                ) in trajectories.items():

                    if len(points) < 2:

                        continue

                    for i in range(
                        1,
                        len(points)
                    ):

                        cv2.line(
                            frame,
                            points[i - 1],
                            points[i],
                            (0, 255, 255),
                            2
                        )

                    current_point = (
                        points[-1]
                    )

                    cv2.circle(
                        frame,
                        current_point,
                        5,
                        (0, 255, 255),
                        -1
                    )

                # -------------------------------------------------
                # FRAME NUMBER
                # -------------------------------------------------

                cv2.putText(
                    frame,
                    f"Frame: {frame_number}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                writer.write(
                    frame
                )

                # -------------------------------------------------
                # PROGRESS
                # -------------------------------------------------

                if frame_number % 10 == 0:

                    progress_value = min(
                        frame_number /
                        max_frames,
                        1.0
                    )

                    progress.progress(
                        progress_value
                    )

                    status.write(
                        f"Processing frame "
                        f"{frame_number}..."
                    )

                if frame_number >= max_frames:

                    break

            writer.release()

            processing_time = (
                time.time() - start_time
            )

            processing_fps = (
                frame_number /
                processing_time
                if processing_time > 0
                else 0
            )

            progress.progress(
                1.0
            )

            status.success(
                f"Tracking completed — "
                f"{frame_number} frames processed."
            )

            # -------------------------------------------------
            # RESULT VIDEO
            # -------------------------------------------------

            st.divider()

            st.subheader(
                "🎯 Tracking Result"
            )

            st.video(
                output_video_path
            )

            # -------------------------------------------------
            # DOWNLOAD VIDEO
            # -------------------------------------------------

            with open(
                output_video_path,
                "rb"
            ) as video_file:

                st.download_button(
                    "⬇️ Download Tracked Video",
                    video_file,
                    file_name=(
                        "visiontrack_tracked.mp4"
                    ),
                    mime="video/mp4",
                    use_container_width=True
                )

            # -------------------------------------------------
            # GET ANALYTICS
            # -------------------------------------------------

            summary = (
                analytics.get_summary()
            )

            class_counts = (
                analytics.get_class_counts()
            )

            total_objects = len(
                summary
            )

            unique_classes = len(
                class_counts
            )

            if summary:

                average_confidence = (
                    sum(
                        obj[
                            "average_confidence"
                        ]
                        for obj in summary
                    )
                    / len(summary)
                )

            else:

                average_confidence = 0

            # -------------------------------------------------
            # PERFORMANCE METRICS
            # -------------------------------------------------

            st.divider()

            st.subheader(
                "⚡ Performance Metrics"
            )

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            with col1:

                st.metric(
                    "Processing FPS",
                    f"{processing_fps:.2f}"
                )

            with col2:

                st.metric(
                    "Processing Time",
                    f"{processing_time:.2f}s"
                )

            with col3:

                st.metric(
                    "Video FPS",
                    f"{fps:.2f}"
                )

            with col4:

                st.metric(
                    "Video Duration",
                    f"{video_duration:.1f}s"
                )

            # -------------------------------------------------
            # TRACKING METRICS
            # -------------------------------------------------

            st.subheader(
                "📊 Tracking Analytics"
            )

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            with col1:

                st.metric(
                    "Tracked Objects",
                    total_objects
                )

            with col2:

                st.metric(
                    "Unique Classes",
                    unique_classes
                )

            with col3:

                st.metric(
                    "Average Confidence",
                    f"{average_confidence * 100:.1f}%"
                )

            with col4:

                st.metric(
                    "Frames Processed",
                    frame_number
                )

            # -------------------------------------------------
            # CLASS DISTRIBUTION
            # -------------------------------------------------

            if class_counts:

                st.markdown(
                    "### 📈 Object Class Distribution"
                )

                chart_data = pd.DataFrame(
                    {
                        "Objects": class_counts
                    }
                )

                st.bar_chart(
                    chart_data
                )

            # -------------------------------------------------
            # TRACKING TABLE
            # -------------------------------------------------

            st.markdown(
                "### 📋 Object Tracking Details"
            )

            if summary:

                table_data = []

                for obj in summary:

                    table_data.append(
                        {
                            "Track ID":
                                obj[
                                    "track_id"
                                ],

                            "Object":
                                obj[
                                    "class"
                                ],

                            "Frames Visible":
                                obj[
                                    "frames_visible"
                                ],

                            "Confidence":
                                (
                                    f"{obj['average_confidence'] * 100:.1f}%"
                                ),

                            "First Frame":
                                obj[
                                    "first_frame"
                                ],

                            "Last Frame":
                                obj[
                                    "last_frame"
                                ],

                            "Movement (pixels)":
                                obj[
                                    "movement_distance"
                                ]
                        }
                    )

                tracking_df = pd.DataFrame(
                    table_data
                )

                st.dataframe(
                    tracking_df,
                    use_container_width=True,
                    hide_index=True
                )

                # -------------------------------------------------
                # CSV
                # -------------------------------------------------

                st.markdown(
                    "### 📥 Export Reports"
                )

                csv_data = (
                    tracking_df.to_csv(
                        index=False
                    )
                )

                st.download_button(
                    "⬇️ Download CSV Report",
                    csv_data,
                    file_name=(
                        "visiontrack_tracking_report.csv"
                    ),
                    mime="text/csv"
                )

                # -------------------------------------------------
                # JSON
                # -------------------------------------------------

                json_data = json.dumps(
                    summary,
                    indent=4
                )

                st.download_button(
                    "⬇️ Download JSON Report",
                    json_data,
                    file_name=(
                        "visiontrack_tracking_report.json"
                    ),
                    mime="application/json"
                )

            # -------------------------------------------------
            # TRAJECTORY SUMMARY
            # -------------------------------------------------

            st.markdown(
                "### 🛤️ Trajectory Summary"
            )

            for (
                track_id,
                points
            ) in trajectories.items():

                if len(points) >= 2:

                    st.write(
                        f"**Track ID {track_id}** "
                        f"— {len(points)} trajectory points"
                    )

    else:

        st.info(
            "👆 Upload a video above to begin "
            "multi-object tracking."
        )


# =========================================================
# LIVE CAMERA
# =========================================================

elif mode == "📡 Live Camera":

    st.subheader(
        "📡 Live Camera Detection"
    )

    st.info(
        "Allow camera access when your browser asks "
        "for permission."
    )

    camera_image = st.camera_input(
        "Capture an image from your camera"
    )

    if camera_image:

        start_time = time.time()

        # -------------------------------------------------
        # SAVE CAMERA IMAGE
        # -------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp_file:

            temp_file.write(
                camera_image.getbuffer()
            )

            camera_path = temp_file.name

        # -------------------------------------------------
        # DETECTION
        # -------------------------------------------------

        with st.spinner(
            "Running YOLO detection..."
        ):

            detector = ObjectDetector()

            result = detector.detect(
                camera_path,
                confidence
            )

            detections = (
                detector.get_detections(
                    result
                )
            )

        processing_time = (
            time.time() - start_time
        )

        # -------------------------------------------------
        # DISPLAY
        # -------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### 📸 Captured Frame"
            )

            st.image(
                camera_image,
                use_container_width=True
            )

        with col2:

            st.markdown(
                "### 🎯 Detection Result"
            )

            annotated_image = (
                result.plot()
            )

            st.image(
                annotated_image,
                channels="BGR",
                use_container_width=True
            )

        st.divider()

        # -------------------------------------------------
        # CAMERA ANALYTICS
        # -------------------------------------------------

        st.subheader(
            "📊 Camera Detection Analytics"
        )

        class_counts = {}

        for detection in detections:

            class_name = (
                detection["class"]
            )

            class_counts[class_name] = (
                class_counts.get(
                    class_name,
                    0
                ) + 1
            )

        total_objects = len(
            detections
        )

        if detections:

            avg_confidence = (
                sum(
                    d["confidence"]
                    for d in detections
                )
                / len(detections)
            )

        else:

            avg_confidence = 0

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        with col1:

            st.metric(
                "Objects Detected",
                total_objects
            )

        with col2:

            st.metric(
                "Unique Classes",
                len(class_counts)
            )

        with col3:

            st.metric(
                "Average Confidence",
                f"{avg_confidence * 100:.2f}%"
            )

        with col4:

            st.metric(
                "Processing Time",
                f"{processing_time:.2f}s"
            )

        # -------------------------------------------------
        # CAMERA CHART
        # -------------------------------------------------

        if class_counts:

            st.markdown(
                "### 📈 Object Distribution"
            )

            chart_data = pd.DataFrame(
                {
                    "Objects": class_counts
                }
            )

            st.bar_chart(
                chart_data
            )

        # -------------------------------------------------
        # CAMERA OBJECT BREAKDOWN
        # -------------------------------------------------

        st.markdown(
            "### 🔎 Detected Objects"
        )

        for class_name, count in (
            class_counts.items()
        ):

            st.write(
                f"**{class_name.title()}** "
                f"— {count} detected"
            )

    else:

        st.warning(
            "Take a picture using the camera "
            "to run object detection."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "VisionTrack AI • YOLO11 • ByteTrack • "
    "Computer Vision • Object Tracking Analytics"
)