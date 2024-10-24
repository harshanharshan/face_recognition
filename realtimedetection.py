import streamlit as st
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Streamlit app layout
st.title("Image Classification with MobileNetV2")
st.write("Upload an image, and the model will classify it.")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process the uploaded image
    img = Image.open(uploaded_file)
    
    # Resize and convert to array
    img_resized = img.resize((224, 224))  # MobileNetV2 expects 224x224 images
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)

    # Display uploaded image
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Make predictions
    try:
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=5)[0]  # Get top 5 predictions

        st.write("Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"{i + 1}: {label} ({score:.2f})")
    except ValueError as e:
        st.error(f"Error in prediction: {str(e)}")

# To run the app, use the command: streamlit run app.py
