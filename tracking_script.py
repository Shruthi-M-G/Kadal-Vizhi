import cv2
import numpy as np

def create_kalman(x, y):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.005
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.2
    kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
    kf.statePost = kf.statePre.copy()
    return kf

selected_point = None
start_frame_idx = 0

def get_click(event, x, y, flags, param):
    global selected_point
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = (x, y)
        print(f"✅ Precision Lock: {x}, {y}")

def on_trackbar(val):
    global start_frame_idx
    start_frame_idx = val

def start_tracking(video_path):
    global selected_point, start_frame_idx
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cv2.namedWindow("Selector")
    cv2.createTrackbar("Frame", "Selector", 0, total_frames - 1, on_trackbar)
    cv2.setMouseCallback("Selector", get_click)
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        ret, frame = cap.read()
        if not ret: break
        display = frame.copy()
        if selected_point: cv2.circle(display, selected_point, 5, (0, 255, 0), -1)
        cv2.imshow("Selector", display)
        if cv2.waitKey(1) & 0xFF == ord('s') and selected_point: break
    
    cv2.destroyWindow("Selector")

    tracker = cv2.TrackerCSRT_create()
    roi = (selected_point[0]-25, selected_point[1]-25, 50, 50)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
    ret, init_frame = cap.read()
    tracker.init(init_frame, roi)
    
    kf = create_kalman(selected_point[0], selected_point[1])
    trajectory = []
    
    
    old_gray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

       
        pred = kf.predict()
        px, py = int(pred[0]), int(pred[1])

       
        success, box = tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in box]
            cx, cy = x + w // 2, y + h // 2
            
      
            
            actual_dist = np.sqrt((cx - px)**2 + (cy - py)**2)
            
            if actual_dist < 60: 
                kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
                target_pos = (cx, cy)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                
                target_pos = (px, py)
                
                tracker.init(frame, (px-w//2, py-h//2, w, h))
        else:
            
            target_pos = (px, py)
            tracker.init(frame, (px-25, py-25, 50, 50))

        trajectory.append(target_pos)

        
        if len(trajectory) > 1:
            
            pts = np.array(trajectory[-100:], np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, (0, 255, 255), 2)

        cv2.circle(frame, target_pos, 4, (0, 0, 255), -1)
        cv2.imshow("Accurate Tracker", frame)
        
        if cv2.waitKey(20) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

start_tracking("D:\\VV\\video.mp4")