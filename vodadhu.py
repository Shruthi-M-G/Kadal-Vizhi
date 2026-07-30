import streamlit as st
import cv2
import tempfile
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates
from collections import deque

st.set_page_config(page_title="Robust Fish Tracker", layout="wide")
st.title("🐟 Click-to-Track Fish")
 

def enhance_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def extract_signature(frame, cx, cy, size=40):
    h, w, _ = frame.shape
    x1 = max(0, cx-size)
    y1 = max(0, cy-size)
    x2 = min(w, cx+size)
    y2 = min(h, cy+size)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [16,16], [0,180,0,256])
    cv2.normalize(hist, hist)
    return hist

def match_signature(frame, signature, pred_center, search=120):
    h, w, _ = frame.shape
    px, py = pred_center
    best_score = -1
    best_center = None

    for dx in range(-search, search, 20):
        for dy in range(-search, search, 20):
            cx = px + dx
            cy = py + dy
            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                continue
            sig = extract_signature(frame, cx, cy)
            if sig is None:
                continue
            score = cv2.compareHist(signature, sig, cv2.HISTCMP_CORREL)
            if score > best_score:
                best_score = score
                best_center = (cx, cy)

    if best_score > 0.6:
        return best_center
    return None


for key in ["active","click","signature"]:
    if key not in st.session_state:
        st.session_state[key] = None

uploaded = st.file_uploader("Upload underwater video", type=["mp4","mov"])

if uploaded:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded.read())
    cap = cv2.VideoCapture(tfile.name)

    if not st.session_state.active:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = st.slider("Select frame & click fish", 0, total-1, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if ret:
            frame = enhance_frame(frame)
            coords = streamlit_image_coordinates(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            if coords:
                cx, cy = int(coords["x"]), int(coords["y"])
                sig = extract_signature(frame, cx, cy)
                if sig is not None:
                    st.session_state.signature = sig
                    st.session_state.click = (cx, cy, idx)
                    st.session_state.active = True
                    st.success("Fish locked ✅")
                    st.rerun()

    
    if st.session_state.active:
        cx, cy, start = st.session_state.click
        tracker = cv2.TrackerCSRT_create()
        last_box = [cx-35, cy-35, 70, 70]
        path = deque(maxlen=800)

        kf = cv2.KalmanFilter(4,2)
        kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        kf.processNoiseCov = np.eye(4,dtype=np.float32)*0.01
        kf.statePre = np.array([[cx],[cy],[0],[0]],np.float32)

        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        frame_box = st.empty()
        f = 0
        init = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = enhance_frame(frame)
            disp = frame.copy()
            pred = kf.predict()
            pred_center = (int(pred[0][0]), int(pred[1][0]))

            if f == start:
                tracker.init(frame, tuple(last_box))
                init = True

            if init:
                ok, box = tracker.update(frame)
                if ok:
                    x,y,w,h = map(int,box)
                    center = (x+w//2,y+h//2)
                else:
                    center = match_signature(frame, st.session_state.signature, pred_center)
                    if center:
                        x = center[0]-35
                        y = center[1]-35
                        w,h = 70,70
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame,(x,y,w,h))
                    else:
                        center = pred_center
                        x,y,w,h = last_box

                kf.correct(np.array([[center[0]],[center[1]]],np.float32))
                last_box = [x,y,w,h]
                path.append(center)

                if len(path)>2:
                    cv2.polylines(disp,[np.array(path,np.int32).reshape(-1,1,2)],False,(0,0,255),2)
                cv2.rectangle(disp,(x,y),(x+w,y+h),(0,255,0),2)

            frame_box.image(cv2.cvtColor(disp,cv2.COLOR_BGR2RGB))
            f+=1

        if st.button("Reset"):
            st.session_state.active=False
            st.session_state.click=None
            st.session_state.signature=None
            st.rerun()