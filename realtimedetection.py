import streamlit as st
import cv2
import numpy as np

# Set title for the Streamlit app
st.title("Real-Time Face Detection")

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Define a function to process the frames
def process_frame(frame):
    # Convert the image from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Here, you can add face detection or other processing code
    # For now, we just return the frame
    return frame

# Create a placeholder for the video feed
frame_placeholder = st.empty()

# Streamlit app loop
while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()
    
    if not ret:
        st.write("Failed to capture video")
        break
    
    # Process the frame
    processed_frame = process_frame(frame)

    # Display the frame
    frame_placeholder.image(processed_frame, channels="RGB", use_column_width=True)

# Release the video capture when done
video_capture.release()
