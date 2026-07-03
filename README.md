# 🏍️ RoadEye ADAS

> **A web application that analyzes road hazards in real-time using a custom-trained deep learning model and dynamic horizon control.**

---

## ⚙️ How It Works

* **Detect:** Upload any road video (or helmet-cam footage) and get instant visual classification (**Path Clear** / **Hazard in Path**) with precise bounding boxes around obstacles like potholes.
* **Alert:** The system acts as an Advanced Rider Assistance System (ADAS), providing asynchronous auditory warnings with a built-in cooldown mechanism when a hazard enters the critical zone.
* **Adjust:** Use the Streamlit sidebar to control detection sensitivity and the **Look Ahead Distance** (Horizon Level) to fit different riding speeds and camera angles.

---

## 📁 Project Structure

```text
ROADEYE/
├── app.py          # Streamlit UI, video processing loop, and audio management
├── src/
│   └── model.py    # Model architecture (Grid-based CNN)
├── models/         # Saved model weights (e.g., roadeye_grid_model_new.pt)
├── alert.wav       # Audio file for hazard warnings
├── .gitignore      # Git ignore rules (excludes models and virtual environments)
└── requirements.txt
```

---

## 🚀 Getting Started

**1. Install dependencies**
```bash
pip install -r requirements.txt
```
*(Or manually: `pip install streamlit opencv-python torch numpy pygame`)*

**2. Run the app**
```bash
streamlit run app.py
```

**3. System Calibration**
Use the sidebar to adjust:
* **Detection Sensitivity:** Filters out low-confidence detections.
* **Look Ahead Distance (Horizon):** Lowers or raises the Region of Interest (ROI) to scan further down the road, mimicking a rider's natural line of sight.

**4. Analyze a video**
Upload a road video (MP4, MOV, AVI) and watch the real-time processing. The system automatically utilizes GPU Half-Precision (FP16) if a CUDA device is detected to maintain high FPS.

---

## 📊 Feedback System

Hazard warnings are generated from two sources:

| Source | When used |
| :--- | :--- |
| **Visual Bounding Boxes (OpenCV)** | Always (when a hazard is within the ROI) |
| **Audio Alerts (Pygame)** | Only when a hazard is detected, governed by a 0.5s cooldown to prevent alert fatigue |

* **Metrics measured:** Processing Speed (Moving Average FPS), Active Hazards count, System Status (Path Clear / Immediate Danger).

---

## 💻 Requirements

* Python 3.8+
* PyTorch
* OpenCV
* Streamlit
* Numpy
* Pygame

---

## ⚠️ Notes

* The audio alert system requires the `alert.wav` file to be present in the root directory. If missing, the app will run silently with a visual-only warning.
* The model weights folder (`models/`) is excluded from version control due to file size limits.