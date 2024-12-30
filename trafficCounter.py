import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import defaultdict

class VehicleTracker:
    def __init__(self, memory_frames=30):
        self.tracks = defaultdict(lambda: {'positions': [], 'counted': False, 'frames_missing': 0})
        self.memory_frames = memory_frames
        self.counted_vehicles = 0

    def update(self, detections, line_y, threshold=10):
        current_ids = set()
        
        if detections and len(detections) > 0 and detections[0].boxes is not None:
            boxes = detections[0].boxes
            
            for box in boxes:
                if not hasattr(box, 'id'):
                    continue
                    
                track_id = int(box.id[0])
                current_ids.add(track_id)
                
                # Calculate center
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_y = (y1 + y2) / 2
                
                # Update track
                track = self.tracks[track_id]
                track['positions'].append(center_y)
                track['frames_missing'] = 0
                
                # Check if vehicle crossed the line
                if not track['counted']:
                    prev_pos = track['positions'][-2] if len(track['positions']) > 1 else center_y
                    if (prev_pos < line_y <= center_y) or (prev_pos > line_y >= center_y):
                        track['counted'] = True
                        self.counted_vehicles += 1
        
        # Update missing frames for tracks not seen in current frame
        for track_id in list(self.tracks.keys()):
            if track_id not in current_ids:
                self.tracks[track_id]['frames_missing'] += 1
                
                # Remove old tracks
                if self.tracks[track_id]['frames_missing'] > self.memory_frames:
                    del self.tracks[track_id]

        return self.counted_vehicles

class VehicleCounter:
    def __init__(self, model_path="yolov8x.pt", confidence=0.3):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.tracker = VehicleTracker()
        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
    def draw_debug(self, frame, detections, line_y):
        height, width = frame.shape[:2]
        
        # Draw counting line
        cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 2)
        
        if detections and len(detections) > 0:
            boxes = detections[0].boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                # Only process vehicle classes
                if cls not in self.vehicle_classes:
                    continue
                
                # Calculate center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                
                # Draw tracking ID if available
                if hasattr(box, 'id'):
                    track_id = int(box.id[0])
                    cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1)-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw class and confidence
                label = f"{self.vehicle_classes[cls]} {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw counter
        cv2.putText(frame, f"Count: {self.tracker.counted_vehicles}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame

    def process_frame(self, frame, line_y):
        # Run detection with tracking
        results = self.model.track(frame, 
                                 conf=self.confidence, 
                                 persist=True,
                                 tracker="bytetrack.yaml",
                                 verbose=False)
        
        # Update vehicle count
        count = self.tracker.update(results, line_y)
        
        # Draw debug visualization
        frame = self.draw_debug(frame, results, line_y)
        
        return frame, count

def main():
    st.set_page_config(page_title="Vehicle Counter", layout="wide")
    
    st.title("Vehicle Counter with ByteTrack")
    
    # Sidebar settings
    st.sidebar.header("Settings")
    confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.3)
    line_position = st.sidebar.slider("Counting Line Position (%)", 0, 100, 50)
    
    # Model selection
    model_option = st.sidebar.selectbox(
        "Select YOLO model",
        ["yolov8m.pt", "yolov8l.pt", "yolov8x.pt", "yolov11x.pt"]
    )
    
    # Debug settings
    show_debug = st.sidebar.checkbox("Show Debug Visualization", True)
    
    video_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
    
    if video_file:
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.read())
        
        try:
            cap = cv2.VideoCapture("temp_video.mp4")
            
            if not cap.isOpened():
                st.error("Error opening video file")
                return
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Initialize counter with selected model
            counter = VehicleCounter(model_path=model_option, confidence=confidence)
            
            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            count_text = st.empty()
            frame_display = st.empty()
            
            line_y = int(height * line_position / 100)
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, current_count = counter.process_frame(frame, line_y)
                
                if show_debug:
                    # Convert for display
                    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_display.image(display_frame)
                
                # Update progress
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")
                count_text.text(f"Vehicles counted: {current_count}")
                
                frame_count += 1
            
            cap.release()
            st.success(f"Video processing complete! Total vehicles counted: {counter.tracker.counted_vehicles}")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Error details:", e.__class__.__name__)

if __name__ == "__main__":
    main()
