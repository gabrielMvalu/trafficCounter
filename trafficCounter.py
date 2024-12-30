import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

def init_tracker():
    """Initialize tracking dictionary"""
    return {
        'tracks': defaultdict(dict),
        'count': 0,
        'crossed_ids': set()
    }

def point_in_line(point, line_start, line_end, threshold=5):
    """Check if a point is near a line segment"""
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    d = abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
    return d < threshold

class VehicleCounter:
    def __init__(self, model_path="yolov8m.pt", confidence=0.1):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.tracker = init_tracker()
        
    def process_frame(self, frame, counting_line):
        """Process a single frame and update vehicle count"""
        results = self.model.track(frame, conf=self.confidence, persist=True)
        
        if results[0].boxes.id is None:
            return frame, 0
        
        boxes = results[0].boxes.xyxy.cpu()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        cls = results[0].boxes.cls.cpu().numpy().astype(int)
        
        current_count = 0
        
        # Draw counting line
        cv2.line(frame, counting_line[0], counting_line[1], (0, 255, 0), 2)
        
        for box, id, cl in zip(boxes, ids, cls):
            # Only count vehicles (customize class IDs as needed)
            if cl not in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                continue
                
            x1, y1, x2, y2 = box.numpy()
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            # Draw bounding box and ID
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {id}", (int(x1), int(y1-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Check if vehicle crossed the line
            if id not in self.tracker['crossed_ids']:
                if point_in_line(center, counting_line[0], counting_line[1]):
                    self.tracker['crossed_ids'].add(id)
                    self.tracker['count'] += 1
                    current_count = 1
                    
        return frame, current_count

def main():
    st.set_page_config(page_title="Vehicle Counter", layout="wide")
    
    st.image("logo.png", width=200)
    st.title("Advanced Vehicle Counter with YOLO")
    
    # Sidebar configurations
    st.sidebar.header("Settings")
    confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.3)
    custom_line_y = st.sidebar.slider("Counting Line Position", 0, 1000, 400)
    
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error reading the video file!")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize counting line
        line_points = [(0, custom_line_y), (width, custom_line_y)]
        
        # Create video writer for saving results
        output_path = "output_video.mp4"
        out = cv2.VideoWriter(output_path, 
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            fps, 
                            (width, height))
        
        # Initialize counter
        counter = VehicleCounter(confidence=confidence)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        count_text = st.empty()
        
        # Process video
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                processed_frame, current_count = counter.process_frame(frame, line_points)
                
                # Update display
                if current_count > 0:
                    count_text.text(f"Total vehicles counted: {counter.tracker['count']}")
                
                # Write frame
                out.write(processed_frame)
                
                # Update progress
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")
                
            # Display final results
            st.success(f"Video processing complete! Total vehicles counted: {counter.tracker['count']}")
            
            # Display processed video
            st.video(output_path)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
        finally:
            cap.release()
            out.release()

if __name__ == "__main__":
    main()
