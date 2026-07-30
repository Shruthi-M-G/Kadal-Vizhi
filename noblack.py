import torch
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import torch.serialization
from ultralytics.nn.tasks import DetectionModel
import functools

# --- PYTORCH SECURITY FIX ---
torch.serialization.add_safe_globals([DetectionModel])
torch.load = functools.partial(torch.load, weights_only=False)

MODEL_PATH = "D:/VV/runs/detect/train/weights/best.pt"
VIDEO_PATH = "D:/VV/video.mp4"

model = YOLO(MODEL_PATH)

selected_id = None
current_boxes = []
trajectories = {} 

def handle_mouse_click(event, x, y, flags, param):
    global selected_id, trajectories
    if event == cv2.EVENT_LBUTTONDOWN:
        for box_data in current_boxes:
            bx, by, bw, bh, tid = box_data
            if (bx - bw/2) < x < (bx + bw/2) and (by - bh/2) < y < (by + bh/2):
                selected_id = tid
                if tid in trajectories:
                    trajectories[tid].clear()
                print(f"Locked ID: {selected_id}")
                break

cv2.namedWindow("Fish Tracker")
cv2.setMouseCallback("Fish Tracker", handle_mouse_click)
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # --- ADVANCED TRACKING FOR TURNING FISH ---
    results = model.track(
        frame, 
        persist=True, 
        tracker="botsort.yaml", # It uses 'track_buffer' internally
        conf=0.1,               # Detect even very small/turned fish
        iou=0.6,                
        imgsz=1280,             
        # vid_stride=1 ensures we don't skip frames during turns
        device='cpu', 
        verbose=False
    )

    if results[0].boxes.id is not None:
        boxes_xywh = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        current_boxes = []
        for i, tid in enumerate(track_ids):
            x_c, y_c, w_h, h_h = boxes_xywh[i]
            current_boxes.append((x_c, y_c, w_h, h_h, tid))
            
            if tid not in trajectories:
                trajectories[tid] = deque(maxlen=2000)
            trajectories[tid].append((int(x_c), int(y_c)))

            if tid == selected_id:
                if len(trajectories[tid]) > 1:
                    pts = np.array(list(trajectories[tid]), np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], isClosed=False, color=(255, 0, 255), thickness=2)
                cv2.circle(frame, (int(x_c), int(y_c)), 5, (255, 0, 255), -1)

    cv2.imshow("Fish Tracker", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('r'): selected_id = None

cap.release()
cv2.destroyAllWindows()