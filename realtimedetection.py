import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to enhance image quality
def enhance_image(image):
    # Convert the image from RGB (PIL) to BGR (OpenCV)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Apply histogram equalization for contrast enhancement
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_eq = cv2.equalizeHist(img_gray)
    
    # Apply a sharpening filter
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img_sharp = cv2.filter2D(img_eq, -1, kernel)
    
    # Denoise the image using Non-Local Means Denoising
    img_denoised = cv2.fastNlMeansDenoising(img_sharp, None, 10, 7, 21)
    
    # Convert back to RGB
    img_enhanced = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_enhanced)

# Streamlit UI
st.title("Image Quality Enhancer")
st.write("Upload an image to enhance its quality.")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    
    # Enhance the image
    enhanced_image = enhance_image(image)
    
    # Display the original and enhanced images
    st.image(image, caption="Original Image", use_column_width=True)
    st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)
