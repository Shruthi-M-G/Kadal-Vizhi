import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Fish Tracker Pro", layout="wide")
st.title("🐠 Fish Trajectory Tracker - Ultimate Fix")

if 'track_point' not in st.session_state:
    st.session_state['track_point'] = None

uploaded_file = st.sidebar.file_uploader("Upload Fish Video", type=["mp4", "mov", "avi"])

def enhance_frame(frame):
    # Underwater contrast booster
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    cap = cv2.VideoCapture("temp_video.mp4")
    ret, first_frame = cap.read()
    
    if ret and st.session_state['track_point'] is None:
        st.subheader("Select a fish by clicking on it:")
        display_frame = enhance_frame(first_frame)
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        value = streamlit_image_coordinates(rgb_frame, key="coords")

        if value:
            st.session_state['track_point'] = (value['x'], value['y'])
            st.rerun()

    if st.session_state['track_point']:
        st_frame = st.empty()
        stop_btn = st.sidebar.button("Stop/Reset")
        
        px, py = st.session_state['track_point']
        
        # --- ROBUST TRACKER CREATION ---
        tracker = None
        # Trying the most modern way (OpenCV 4.5.1 to 5.0)
        try:
            tracker = cv2.TrackerCSRT.create()
        except AttributeError:
            # If the above fails, try other known methods
            methods = [
                lambda: cv2.TrackerCSRT_create(),
                lambda: cv2.legacy.TrackerCSRT_create(),
                lambda: cv2.TrackerKCF.create() # Fallback to KCF if CSRT is missing
            ]
            for method in methods:
                try:
                    tracker = method()
                    break
                except AttributeError:
                    continue
        
        if tracker is None:
            st.error("OpenCV Tracking modules not found! Run: pip install opencv-contrib-python")
            st.stop()

        # Defining ROI
        roi = (int(px - 35), int(py - 35), 70, 70)
        tracker.init(first_frame, roi)
        
        trajectory = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_btn:
                break
            
            # Enhance for accuracy
            enhanced = enhance_frame(frame)
            success, box = tracker.update(enhanced)
            
            if success:
                x, y, w, h = [int(v) for v in box]
                center = (x + w//2, y + h//2)
                trajectory.append(center)
                
                # Draw Trajectory
                if len(trajectory) > 1:
                    pts = np.array(trajectory, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], isClosed=False, color=(128, 0, 128), thickness=3)
                
                cv2.circle(frame, center, 6, (0, 255, 0), -1)
            else:
                cv2.putText(frame, "LOST TRACKING", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
        if stop_btn:
            st.session_state['track_point'] = None
            st.rerun()

    cap.release()