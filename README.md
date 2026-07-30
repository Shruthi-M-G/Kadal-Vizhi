# 🐠 Kadal-Vizhi (கடல் விழி)
> **Interactive Underwater Fish Detection & Motion Tracking using YOLOv8 and BoT-SORT**

`Kadal-Vizhi` is an AI-powered computer vision system designed to detect underwater marine species and interactively track their movement trajectories in real-time. By fine-tuning YOLOv8 on custom underwater datasets and leveraging BoT-SORT, the system addresses key challenges like low visibility, target occlusion, and rapid aquatic motion.

---

## Key Features
- **Custom Fine-Tuned Model**: Fine-tuned YOLOv8 on marine datasets to handle degraded visibility and ambient light shifts underwater.
- **Interactive Target Locking**: Real-time OpenCV click-to-lock feature that allows users to isolate and track a specific fish's movement path.
- **Dynamic Path Trajectory**: Visualizes long-range motion history for selected marine organisms.
- **Robust Multi-Object Tracking**: Integrates BoT-SORT to prevent ID switching during object overlapping and tight turning maneuvers.

---

## Project Structure
```text
Kadal-Vizhi/
├── dataset/                  # Marine images (train/val) & bounding box annotations
├── runs/                     # Model metrics & trained weights (best.pt)
├── supervised.py             # Main entry point: Interactive GUI tracking pipeline
├── train.py                  # Model fine-tuning configuration script
├── custom_tracker.yaml       # BoT-SORT tracking hyperparameter settings
├── data.yaml                 # YOLO dataset path configurations
├── video.mp4                 # Sample input underwater stream
└── README.md
