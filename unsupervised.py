import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates
import time

st.set_page_config(page_title="Fish Tracker", layout="wide")
st.title("🐠 Click on a Fish to Track")

# -------------------------
# Session State Setup
# -------------------------
if "track_point" not in st.session_state:
    st.session_state.track_point = None

if "tracking" not in st.session_state:
    st.session_state.tracking = False

uploaded_file = st.sidebar.file_uploader(
    "Upload Fish Video", type=["mp4", "mov", "avi"]
)

# -------------------------
# If video uploaded
# -------------------------
if uploaded_file:

    # Save uploaded video
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())

    cap = cv2.VideoCapture("temp_video.mp4")
    ret, first_frame = cap.read()

    if not ret:
        st.error("Error reading video file.")
        st.stop()

    height, width = first_frame.shape[:2]

    # -------------------------
    # FISH SELECTION STAGE
    # -------------------------
    if st.session_state.track_point is None:

        st.subheader("Select the fish to track")

        rgb_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        value = streamlit_image_coordinates(rgb_frame, key="coords")

        if value:
            x = int(value["x"])
            y = int(value["y"])

            box_size = 40

            # Safe bounding box
            x1 = max(0, x - box_size // 2)
            y1 = max(0, y - box_size // 2)
            w = min(box_size, width - x1)
            h = min(box_size, height - y1)

            st.session_state.track_point = (x1, y1, w, h)
            st.session_state.tracking = True
            st.rerun()

    # -------------------------
    # TRACKING STAGE
    # -------------------------
    if st.session_state.tracking:

        st_frame = st.empty()
        stop_btn = st.sidebar.button("Stop / Reset")

        tracker = cv2.TrackerCSRT_create()
        tracker.init(first_frame, st.session_state.track_point)

        trajectory = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while cap.isOpened():

            if stop_btn:
                break

            ret, frame = cap.read()
            if not ret:
                break

            success, box = tracker.update(frame)

            # 🔴 STOP if fish disappears
            if not success:
                st.warning("⚠ Fish disappeared! Tracking stopped.")
                break

            x, y, w, h = [int(v) for v in box]

            # Extra safety check
            if w <= 0 or h <= 0:
                st.warning("⚠ Fish disappeared! Tracking stopped.")
                break

            center = (x + w // 2, y + h // 2)
            trajectory.append(center)

            # Draw trajectory path
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i - 1],
                         trajectory[i], (128, 0, 128), 2)

            # Draw center dot
            cv2.circle(frame, center, 4, (128, 0, 128), -1)

            st_frame.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

            time.sleep(0.03)

        cap.release()

        # Reset if stopped
        if stop_btn or not success:
            st.session_state.track_point = None
            st.session_state.tracking = False
            st.rerun()