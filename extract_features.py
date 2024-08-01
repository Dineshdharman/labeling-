import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import logging

# Configure logging without specifying encoding to avoid UnicodeEncodeError
logging.basicConfig(filename='feature_extraction.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

# Set environment variable to handle encoding issues
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Suppress TensorFlow warnings and output
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Check TensorFlow and Keras versions
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# Load the VGG16 model pre-trained on ImageNet
base_model = VGG16(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

# Function to extract features
def extract_features(img_path, model):
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        
        # Check preprocessed image data
        if img_data is None or img_data.size == 0:
            raise ValueError(f"Image data is empty for {img_path}")
        
        logging.debug(f"Preprocessed image data shape: {img_data.shape}")
        
        # Predict features
        features = model.predict(img_data)
        
        # Log features
        logging.debug(f"Image path: {img_path}")
        logging.debug(f"Features shape: {features.shape}")
        logging.debug(f"Sample features: {features.flatten()[:10]}")

        feature_array = features.flatten()
        
        if np.all(feature_array == 0):
            logging.warning(f"Warning: Features for {img_path} are all zeros.")
        
        return feature_array
    except Exception as e:
        logging.error(f"Error extracting features from {img_path}: {e}")
        return np.zeros((4096,))  # Adjust size as needed

# Directory containing the images
image_dir = r'C:\Users\HP\Desktop\model\uploads'

# Extract features for labeled images
df = pd.read_excel('labeled_images.xlsx')
features_list = []

for index, row in df.iterrows():
    img_path = os.path.join(image_dir, row['Image'])
    if os.path.exists(img_path):
        try:
            features = extract_features(img_path, model)
            features_list.append(features)
        except Exception as e:
            logging.error(f"Error processing image {img_path}: {e}")
            features_list.append(np.zeros((4096,)))  # Append zeros if there's an error
    else:
        logging.error(f"Image not found: {img_path}")
        features_list.append(np.zeros((4096,)))  # Append zeros if the image is not found

# Add features to the DataFrame
features_df = pd.DataFrame(features_list)
logging.debug(f"Features DataFrame head:\n{features_df.head()}")

# Save features to a text file for debugging
with open('features_debug.txt', 'w', encoding='utf-8') as f:
    for features in features_list:
        f.write(','.join(map(str, features)) + '\n')

# Combine original DataFrame with features DataFrame
labeled_df = pd.concat([df, features_df], axis=1)

# Save DataFrame to Excel file using openpyxl engine
excel_path = 'labeled_images_with_features.xlsx'
try:
    labeled_df.to_excel(excel_path, index=False, engine='openpyxl')
    logging.info(f"Excel file saved successfully at {excel_path}.")
except PermissionError as e:
    logging.error(f"Permission error while saving Excel file: {e}")
    print(f"Permission error while saving Excel file: {e}")
except Exception as e:
    logging.error(f"Error saving Excel file: {e}")
    print(f"Error saving Excel file: {e}")

# Function to preprocess and inspect a sample image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return img_data

# Standalone test of model prediction with sample image
sample_img_path = r'C:\Users\HP\Desktop\model\Dataset\2220.jpg'
sample_img_data = preprocess_image(sample_img_path)
print(f"Preprocessed image data shape: {sample_img_data.shape}")
print(f"Sample preprocessed image data: {sample_img_data}")

# Test model prediction on sample image
features = extract_features(sample_img_path, model)
print(f"Extracted features: {features.flatten()[:10]}")
