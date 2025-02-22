import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from scipy.spatial import distance

class VehicleCounter:
    def __init__(self, model_path="yolov10n.pt", confidence=0.25):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.vehicle_count = 0
        self.tracked_vehicles = {}
        self.frame_id = 0
        # Clase pentru vehicule în YOLO
        self.vehicle_classes = {
            2: 'car', 
            3: 'motorcycle', 
            5: 'bus', 
            7: 'truck'
        }

    def draw_debug_info(self, frame, detections, line_y1, line_y2):
        """Desenează informații de debugging pe frame"""
        height, width = frame.shape[:2]
        
        # Desenează liniile de numărare
        cv2.line(frame, (0, line_y1), (width, line_y1), (255, 0, 0), 2)
        cv2.putText(frame, "Counting Line 1", (10, line_y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.line(frame, (0, line_y2), (width, line_y2), (0, 255, 0), 2)
        cv2.putText(frame, "Counting Line 2", (10, line_y2 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Desenează toate detecțiile
        if detections:
            boxes = detections[0].boxes
            for box in boxes:
                # Extrage coordonatele și clasa
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                # Verifică dacă este vehicul
                if cls in self.vehicle_classes:
                    color = (0, 255, 0)  # Verde pentru vehicule
                else:
                    color = (0, 0, 255)  # Roșu pentru alte obiecte
                
                # Desenează bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Adaugă text cu clasa și confidence
                label = f"{self.vehicle_classes.get(cls, f'Class {cls}')} {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Adaugă contor în colțul frame-ului
        cv2.putText(frame, f"Count: {self.vehicle_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame

    def process_frame(self, frame, line_y1, line_y2):
        """Procesează un frame și numără vehiculele"""
        # Rulează detectia YOLO
        results = self.model(frame, conf=self.confidence, verbose=False)
        
        height = frame.shape[0]
        buffer_zone = 10  # Pixeli în jurul liniei
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            current_frame_objects = []

            for box in boxes:
                # Verifică clasa
                cls = int(box.cls[0].cpu().numpy())
                if cls not in self.vehicle_classes:
                    continue
                
                # Calculează centrul obiectului
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                current_frame_objects.append((center_x, center_y))

                # Verifică dacă obiectul traversează liniile în ordine
                if line_y1 - buffer_zone <= center_y <= line_y1 + buffer_zone:
                    matched = False

                    for tracked_id, (tracked_x, tracked_y, passed_line2) in self.tracked_vehicles.items():
                        if distance.euclidean((center_x, center_y), (tracked_x, tracked_y)) < 20:
                            matched = True
                            self.tracked_vehicles[tracked_id] = (center_x, center_y, passed_line2)
                            break

                    if not matched:
                        self.tracked_vehicles[self.frame_id] = (center_x, center_y, False)
                        self.frame_id += 1

                for tracked_id, (tracked_x, tracked_y, passed_line2) in list(self.tracked_vehicles.items()):
                    if not passed_line2 and line_y2 - buffer_zone <= center_y <= line_y2 + buffer_zone:
                        self.tracked_vehicles[tracked_id] = (center_x, center_y, True)
                        self.vehicle_count += 1

            # Elimină vehiculele care nu au fost văzute în ultimele 50 de cadre
            self.tracked_vehicles = {k: v for k, v in self.tracked_vehicles.items() if k > self.frame_id - 50}
        
        # Adaugă informații de debugging
        frame = self.draw_debug_info(frame, results, line_y1, line_y2)
        
        return frame

def main():
    st.set_page_config(page_title="Vehicle Counter", layout="wide")
    
    st.title("Vehicle Counter with YOLO - Dual Line Counting")
    
    # Configurări în sidebar
    st.sidebar.header("Settings")
    confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.25)
    line_position1 = st.sidebar.slider("Counting Line 1 Position (%)", 0, 100, 40)
    line_position2 = st.sidebar.slider("Counting Line 2 Position (%)", 0, 100, 60)
    
    # Model selection
    model_option = st.sidebar.selectbox(
        "Select YOLO model",
        ["yolov10n.pt", "yolov10m.pt", "yolov10l.pt", "yolov10x.pt"]
    )
    
    # Upload video
    video_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
    
    if video_file:
        # Save uploaded video
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.read())
        
        try:
            # Initialize video capture
            cap = cv2.VideoCapture("temp_video.mp4")
            
            if not cap.isOpened():
                st.error("Error opening video file")
                return
            
            # Get video info
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Initialize counter
            counter = VehicleCounter(model_path=model_option, confidence=confidence)
            
            # Create progress indicators
            progress_bar = st.progress(0)
            frame_text = st.empty()
            count_text = st.empty()
            debug_text = st.empty()
            
            # Process video
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate counting line positions
            line_y1 = int(height * line_position1 / 100)
            line_y2 = int(height * line_position2 / 100)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = counter.process_frame(frame, line_y1, line_y2)
                
                # Convert pentru display
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Update progress
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                frame_text.text(f"Processing frame {frame_count}/{total_frames}")
                count_text.text(f"Vehicles counted: {counter.vehicle_count}")
                
                # Display frame
                st.image(processed_frame, channels="RGB", use_container_width=True)
                
                frame_count += 1
            
            cap.release()
            st.success(f"Video processing complete! Total vehicles counted: {counter.vehicle_count}")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Error details:", e.__class__.__name__)
            
if __name__ == "__main__":
    main()
