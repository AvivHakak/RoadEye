import streamlit as st
import cv2
import torch
import numpy as np
from src.model import RoadEyeGridModel
import tempfile
import time
from collections import deque
import pygame

st.set_page_config(page_title="RoadEye", page_icon="🏍️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
    padding-left: 1rem;
    padding-right: 1rem;
    max-width: 100%;
}
.stApp {
    background-color: #F8FAFC;
    color: #0F172A;
    overflow: hidden;
}
.video-container img {
    max-height: 85vh !important;
    width: auto !important;
    margin: 0 auto;
    display: block;
    object-fit: contain;
    border-radius: 8px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border: 1px solid #CBD5E1;
}
.hazard-warning {
    background-color: #FEF2F2;
    border-left: 6px solid #EF4444;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
}
.hazard-title {
    color: #991B1B;
    font-weight: 800;
    font-size: 1.4rem;
    margin-bottom: 5px;
}
.hazard-type {
    color: #B91C1C;
    font-size: 1.8rem;
    font-weight: 900;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

pygame.mixer.init()

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RoadEyeGridModel()
    model.load_state_dict(torch.load('models/roadeye_grid_model_new.pt', map_location=device))
    if device.type == 'cuda':
        model.half()
    model.to(device)
    model.eval()
    return model, device

def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection_area / float(box1_area + box2_area - intersection_area + 1e-6)

def non_max_suppression(boxes, probs, threshold=0.1):
    if not boxes:
        return [], []
    indices = np.argsort(probs)[::-1]
    keep_boxes = []
    keep_probs = []
    while len(indices) > 0:
        current = indices[0]
        keep_boxes.append(boxes[current])
        keep_probs.append(probs[current])
        if len(indices) == 1:
            break
        remaining_indices = []
        for i in indices[1:]:
            if calculate_iou(boxes[current], boxes[i]) < threshold:
                remaining_indices.append(i)
        indices = remaining_indices
    return keep_boxes, keep_probs

class BoxTracker:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.prev_boxes = []
    def update(self, current_boxes):
        if not self.prev_boxes:
            self.prev_boxes = current_boxes
            return current_boxes
        smoothed_boxes = []
        for curr_box in current_boxes:
            best_iou = 0
            best_prev_box = None
            for prev_box in self.prev_boxes:
                iou = calculate_iou(curr_box, prev_box)
                if iou > best_iou:
                    best_iou = iou
                    best_prev_box = prev_box
            if best_iou > 0.2:
                smoothed_box = [
                    int(self.alpha * curr_box[0] + (1 - self.alpha) * best_prev_box[0]),
                    int(self.alpha * curr_box[1] + (1 - self.alpha) * best_prev_box[1]),
                    int(self.alpha * curr_box[2] + (1 - self.alpha) * best_prev_box[2]),
                    int(self.alpha * curr_box[3] + (1 - self.alpha) * best_prev_box[3])
                ]
                smoothed_boxes.append(smoothed_box)
            else:
                smoothed_boxes.append(curr_box)
        self.prev_boxes = smoothed_boxes
        return smoothed_boxes

def process_frame(frame, model, device, tracker, conf_threshold, horizon_level):
    h_orig, w_orig, _ = frame.shape
    resized_frame = cv2.resize(frame, (224, 224))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    input_tensor = torch.from_numpy(rgb_frame).float() / 255.0
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    if device.type == 'cuda':
        input_tensor = input_tensor.half()
    
    with torch.no_grad():
        out = model(input_tensor)
        
    all_boxes = []
    all_probs = []
    
    x_min_roi, x_max_roi = int(w_orig * 0.35), int(w_orig * 0.65)
    y_min_roi = int(h_orig * horizon_level)
    
    detected_type = None

    for i in range(7):
        for j in range(7):
            prob = out[0, 0, i, j].item()
            if prob > conf_threshold:
                x_norm, y_norm = out[0, 1, i, j].item(), out[0, 2, i, j].item()
                w_norm, h_norm = out[0, 3, i, j].item(), out[0, 4, i, j].item()
                
                x_center, y_center = x_norm * w_orig, y_norm * h_orig
                
                if not (x_min_roi < x_center < x_max_roi and y_center > y_min_roi):
                    continue
                if w_norm > 0.30 or h_norm > 0.20:
                    continue
                
                box_w, box_h = w_norm * w_orig, h_norm * h_orig
                x1, y1 = int(x_center - box_w / 2), int(y_center - box_h / 2)
                x2, y2 = int(x_center + box_w / 2), int(y_center + box_h / 2)
                
                all_boxes.append([x1, y1, x2, y2])
                all_probs.append(prob)
                
                detected_type = "POTHOLE" 

    final_boxes, final_probs = non_max_suppression(all_boxes, all_probs, threshold=0.1)
    smoothed_boxes = tracker.update(final_boxes)
    
    hazard_detected = len(smoothed_boxes) > 0
    if not hazard_detected:
        detected_type = None

    for box, p in zip(smoothed_boxes, final_probs):
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
        cv2.putText(frame, f"Hazard: {p:.2f}", (box[0], box[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    return frame, hazard_detected, detected_type

model, device = load_model()

with st.sidebar:
    st.markdown("### ⚙️ System Settings")
    conf_threshold = st.slider("Detection Sensitivity", min_value=0.30, max_value=0.90, value=0.65, step=0.05)
    horizon_level = st.slider("Look Ahead Distance (Horizon)", min_value=0.15, max_value=0.60, value=0.35, step=0.05)

col1, col2 = st.columns([1, 3])

with col1:
    st.image("https://images.unsplash.com/photo-1558980394-0a37b3614ce8?q=80&w=800&auto=format&fit=crop", use_container_width=True)
    st.markdown("### RoadEye ADAS")
    st.markdown("Advanced Rider Assistance System")
    st.write("---")
    uploaded_video = st.file_uploader("", type=['mp4', 'mov', 'avi'])
    
    warning_ui = st.empty()

with col2:
    metric_cols = st.columns(3)
    fps_metric = metric_cols[0].empty()
    hazards_metric = metric_cols[1].empty()
    status_metric = metric_cols[2].empty()
    
    stframe = st.empty()

if uploaded_video is not None:
    try:
        alert_sound = pygame.mixer.Sound('alert.wav')
    except Exception as e:
        alert_sound = None
        st.sidebar.error("⚠️ alert.wav file not found. System will operate without audio alerts.")

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)
    tracker = BoxTracker(alpha=0.4)
    fps_history = deque(maxlen=10)
    frame_count = 0
    
    last_alert_time = 0
    alert_cooldown = 0.5 
    
    while cap.isOpened():
        start_time = time.time()
        frame_count += 1
        ret, frame = cap.read()
        if not ret: break
            
        processed_frame, is_hazard, obstacle_label = process_frame(frame, model, device, tracker, conf_threshold, horizon_level)
        processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
        fps_metric.metric("Processing Speed", f"{avg_fps:.1f} FPS")
        
        if is_hazard:
            hazards_metric.metric("Active Hazards", "⚠️ Detected")
            status_metric.metric("System Status", "🚨 HAZARD IN PATH!")
            warning_ui.markdown(f"""
                <div class="hazard-warning">
                    <div class="hazard-title">⚠️ IMMEDIATE DANGER</div>
                    <div class="hazard-type">{obstacle_label}</div>
                </div>
            """, unsafe_allow_html=True)
            
            current_time = time.time()
            if current_time - last_alert_time >= alert_cooldown:
                if alert_sound:
                    alert_sound.play()
                last_alert_time = current_time 
                
        else:
            hazards_metric.metric("Active Hazards", "0")
            status_metric.metric("System Status", "✅ PATH CLEAR")
            warning_ui.empty()
            
        if frame_count % 2 == 0:
            stframe.image(processed_rgb, channels="RGB", use_container_width=True)
        
        fps_history.append(1 / (time.time() - start_time + 1e-6))
            
    cap.release()