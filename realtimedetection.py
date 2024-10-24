import streamlit as st
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Define the number of classes in the model (for MobileNetV2, this is 1000)
num_classes = 1000

# Example mapping for some classes (You can extend this)
class_names = {
    0: "tench, Tinca tinca",
    1: "goldfish, Carassius auratus",
    2: "great white shark, white shark, man-eater, man-eater shark",
    3: "tiger shark, Galeocerdo cuvieri",
    4: "hammerhead, hammerhead shark",
    # ...
    594: "sorrel",
    # Extend this dictionary with more class names as needed
}

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
    
    # Get the predicted class index
    class_idx = np.argmax(predictions[0])  # Get the index of the class with the highest score
    confidence = predictions[0][class_idx]  # Get the confidence of the prediction

    # Map class index to class name
    predicted_class_name = class_names.get(class_idx, "Unknown class")

    # Display the results
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write(f"Prediction: {predicted_class_name} (Index: {class_idx}) with confidence: {confidence:.2f}")
