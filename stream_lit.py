import streamlit as st
import numpy as np
import cv2
import io

from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_model1 = Xception(
    include_top=False,
    weights='imagenet',
    input_shape=(331, 331, 3)
)

base_model1.trainable = False

# Create the classification model
model = Sequential([
    base_model1,
    BatchNormalization(renorm=True),
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])

model.load_weights('execption_kolam.h5') 

def perform_classifcation(uploaded_file):
    img = io.BytesIO(uploaded_file.read())
    img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), -1)  # Read the image as a NumPy array

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (331, 331))
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)

    class_indices = np.argmax(predictions, axis=1)
    class_names =['Rangoli', 'Foot','Geometric', 'Strip','Swastik', ] 
    predicted_class = class_names[class_indices[0]]
    return predicted_class

st.title("Rangoli Classification")
st.markdown("Upload an image or capture one using your camera for classification.")

# Create a file upload widget
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png",])

# Create a camera capture button
if st.button("Capture from Camera"):
    st.warning("Camera capture is not supported in this demo.")

# Placeholder for image classification result
classification_result = None

# Perform image classification if an image is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    res = perform_classifcation(uploaded_file)

    # Placeholder for the classification result (replace with actual classification logic)
    classification_result = f"Rangoli class: {res}"

# Display the classification result if available
if classification_result:
    st.success(classification_result)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #8699aa;
    }
    .stTitle {
        color: #fff;
        font-size: 40px;
        margin-bottom: 25px;
    }
    .stMarkdown {
        font-size: 22px;
        color: #333;
    }
    .stButton {
        background-color: #fff;
        color: #fff;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 18px;
        cursor: pointer;
    }
    .stButton:hover {
        background-color: #fff;
    }
    .stSuccess {
        background-color: #A0D468;
        color: #fff;
        padding: 10px;
        border-radius: 5px;
        font-size: 50px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)