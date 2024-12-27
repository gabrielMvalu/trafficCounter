import streamlit as st
import cv2

# Display a logo
st.image("logo.png", width=200)

# Application title
st.title("Vehicle Counter with YOLO11")

# Upload a video file
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Check if a file has been uploaded
if uploaded_video:
    # Save the uploaded video locally for processing
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    # Open the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error reading the video file!")
    else:
        # Get video properties: width, height, and fps
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Define a line of interest for counting vehicles
        line_points = [(20, 400), (1080, 400)]

        try:
            # Import and configure the YOLO model for counting
            from ultralytics import YOLO

            # Initialize the YOLO model for counting
            model = YOLO("yolo11n.pt")

            # Initialize the vehicle counter
            count = 0

            # Iterate through each frame of the video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process the frame and count the vehicles
                results = model.predict(frame)
                for result in results:
                    count += len(result.boxes)  # Count all detected objects

            # Display the final result to the user
            st.success(f"Total number of vehicles: {count}")
        except ImportError:
            # Display an error message if the required library is missing
            st.error("Ultralytics library is not installed or not configured properly. Check requirements.txt and restart the application.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

        # Release resources used by OpenCV
        cap.release()
