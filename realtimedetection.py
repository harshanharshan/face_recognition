import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
from PIL import Image
import io

# Load and preprocess the CIFAR-10 dataset
def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

# Create the CNN model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((32, 32))
    image_array = np.array(image)
    return image_array.astype('float32') / 255.0

# Load the data and create the model
x_train, y_train, x_test, y_test = load_data()
model = create_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model (this may take some time, consider loading a pre-trained model instead)
if st.button('Train Model'):
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
    st.success("Model trained successfully!")

# Streamlit app layout
st.title("CIFAR-10 Image Classification")
st.write("Upload an image of a CIFAR-10 object to classify it.")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    image_array = preprocess_image(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Make predictions
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)

    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    st.write(f"Predicted class: {class_names[predicted_class]}")

# To run the app, use the command: streamlit run app.py
