import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image, ImageFile
import os
import zipfile
from time import sleep

# Ensure PIL handles truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Ensure Kaggle credentials
KAGGLE_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".kaggle")
if not os.path.exists(KAGGLE_CONFIG_DIR):
    os.makedirs(KAGGLE_CONFIG_DIR)
    st.error("Kaggle API credentials are not set up. Please upload your kaggle.json file.")
    kaggle_json = st.file_uploader("Upload kaggle.json", type="json")
    if kaggle_json is not None:
        with open(os.path.join(KAGGLE_CONFIG_DIR, "kaggle.json"), "wb") as f:
            f.write(kaggle_json.getbuffer())
        st.success("kaggle.json uploaded successfully! Reload the app.")
        st.stop()

# Download and extract the dataset
def download_and_extract_dataset():
    """Download and extract the Disney dataset."""
    dataset_path = "disney-characters-dataset.zip"
    if not os.path.exists(dataset_path):
        st.write("Downloading dataset...")
        os.system("kaggle datasets download -d sayehkargari/disney-characters-dataset")
    
    extract_path = "cartoon"
    if not os.path.exists(extract_path):
        st.write("Extracting dataset...")
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(".")

# Function to validate and clean dataset
def validate_and_clean_dataset(directory):
    """Check all images in the directory and remove corrupted ones."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            try:
                with Image.open(os.path.join(root, file)) as img:
                    img.verify()  # Verify the integrity of the image
            except Exception as e:
                st.write(f"Removing corrupted file: {os.path.join(root, file)}")
                os.remove(os.path.join(root, file))

# Call the function to download and extract the dataset
st.write("Setting up dataset...")
download_and_extract_dataset()

# Define paths
train_path = "cartoon/train"
test_path = "cartoon/test"

# Validate datasets
st.write("Validating and cleaning the dataset...")
validate_and_clean_dataset(train_path)
validate_and_clean_dataset(test_path)

# Define the model creation function
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 320, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Preprocess the dataset
img_height = 180
img_width = 320
batch_size = 30

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# Train the model
st.title("Training the Model")
st.write("Training is in progress. Please wait...")
progress_bar = st.progress(0)

model = create_model()

for epoch in range(5):  # Reduced epochs for quicker training in the app
    history = model.fit(
        train_generator,
        epochs=1,
        validation_data=test_generator,
        verbose=1
    )
    progress_bar.progress((epoch + 1) / 5)
    sleep(0.5)  # Simulating training delay for visualization

st.success("Model training complete!")

# Define class names
class_names = ['olaf', 'pumba']

# Streamlit app
st.title("Disney Character Classification")
st.write("Upload an image of a Disney character to classify it as either Olaf or Pumba.")

# Upload image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = Image.open(uploaded_file).resize((320, 180))  # Resize to model input size
    image_array = img_to_array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict the class
    st.write("Classifying the image... Please wait.")
    with st.spinner("Running the model..."):
        result = model.predict(image_array)
        predicted_label = class_names[int(result[0] > 0.5)]  # Sigmoid output thresholded at 0.5

    # Display the result
    st.success(f"The model predicts: **{predicted_label}**")
