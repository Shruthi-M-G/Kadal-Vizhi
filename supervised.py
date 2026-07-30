import os
import ssl

# --- SSL VERIFICATION BYPASS (Fixes urllib3/SSL crash) ---
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'

import torch
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import functools

# --- PyTorch 2.6+ Security Bypass ---
torch.serialization.add_safe_globals([DetectionModel])
torch.load = functools.partial(torch.load, weights_only=False)

# --- GLOBAL VARIABLES FOR INTERACTIVE TRACKING ---
selected_id = None
current_boxes = []

def select_fish(event, x, y, flags, param):
    global selected_id, current_boxes
    if event == cv2.EVENT_LBUTTONDOWN:
        for box_data in current_boxes:
            bx, by, bw, bh, tid = box_data
            # Check if click is inside the bounding box
            if (bx - bw/2) < x < (bx + bw/2) and (by - bh/2) < y < (by + bh/2):
                selected_id = tid
                print(f"✅ Target Locked! Tracking Fish ID: {selected_id}")
                break

# --- DYNAMIC WEIGHTS & VIDEO PATH CONFIG ---
# Folder structure check panni weights pick pannum
MODEL_PATH = "runs/detect/train/weights/best.pt"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "dataset/runs/detect/fish_tracking2/weights/best.pt"

VIDEO_PATH = "video.mp4"

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Trained weights file illai in: {MODEL_PATH}")
    exit()

if not os.path.exists(VIDEO_PATH):
    print(f"❌ Error: {VIDEO_PATH} file root folder-la illai!")
    exit()

print(f"Loading Model: {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
track_history = {}
MAX_HISTORY = 200  # Trajectory length

cap = cv2.VideoCapture(VIDEO_PATH)
cv2.namedWindow("Click a Fish to Track")
cv2.setMouseCallback("Click a Fish to Track", select_fish)

print("\n--- INSTRUCTIONS ---")
print("1. Click on any fish to lock & track its path.")
print("2. Press 'r' on keyboard to reset target.")
print("3. Press 'q' to exit.\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        break

    # BoT-SORT tracker execution
    results = model.track(frame, persist=True, tracker="botsort.yaml", device='cpu', verbose=False)

    if results[0].boxes.id is not None:
        boxes_xywh = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        # Current frame boxes update
        current_boxes = []
        for i, tid in enumerate(track_ids):
            x_c, y_c, w_h, h_h = boxes_xywh[i]
            current_boxes.append((x_c, y_c, w_h, h_h, tid))

        for box, tid in zip(boxes_xywh, track_ids):
            # Update history queue
            track = track_history.get(tid, deque(maxlen=MAX_HISTORY))
            track.append((float(box[0]), float(box[1])))
            track_history[tid] = track

            # Draw trajectory only for selected target ID
            if selected_id is not None and tid == selected_id:
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                
                # Yellow path line
                cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 255), thickness=3)
                
                # Green Bounding Box & Label
                cv2.rectangle(frame, (int(box[0]-box[2]/2), int(box[1]-box[3]/2)), 
                              (int(box[0]+box[2]/2), int(box[1]+box[3]/2)), (0, 255, 0), 2)
                cv2.putText(frame, f"TARGET {tid}", (int(box[0]-box[2]/2), int(box[1]-box[3]/2)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display prompt on frame if nothing is selected
    if selected_id is None:
        cv2.putText(frame, "CLICK A FISH TO LOCK TARGET", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Click a Fish to Track", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): 
        break
    if key == ord("r"): 
        selected_id = None
        print("🔄 Selection Reset")

cap.release()
cv2.destroyAllWindows()