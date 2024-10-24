import streamlit as st
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Define the number of classes in the model (for MobileNetV2, this is 1000)
num_classes = 1000

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = load_img(image, target_size=(224, 224))  # Resize the image to 224x224
    img_array = img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# Streamlit UI
st.title("Image Classification with MobileNetV2")
st.write("Upload an image, and the model will classify it.")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the image
    img_array = preprocess_image(uploaded_file)
    
    # Make predictions
    predictions = model.predict(img_array)
    
    # Decode the predictions to get class labels
    decoded_predictions = model.predict(img_array)
    class_idx = np.argmax(decoded_predictions[0])  # Get the index of the class with the highest score
    confidence = decoded_predictions[0][class_idx]  # Get the confidence of the prediction

    # Display the results
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write(f"Prediction: {class_idx} with confidence: {confidence:.2f}")

    # You can map class_idx to class names if needed
    # class_names = {0: 'Class1', 1: 'Class2', ...}
    # st.write(f"Predicted Class: {class_names[class_idx]}")

