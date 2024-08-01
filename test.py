import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bar

# Function to preprocess the image
def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        return img_data
    except Exception as e:
        print(f"Error preprocessing image {img_path}: {e}")
        return None

# Function to extract features using ResNet50
def extract_features(img_data, model):
    try:
        features = model.predict(img_data)
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Load the ResNet50 model pretrained on ImageNet
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)  # Remove the last layer to get features

# Directory containing your images
image_dir = r'C:\Users\HP\Desktop\model\uploads'

# List to hold features and image paths
data = []

# Process each image in the directory
image_files = [os.path.join(image_dir, img_file) for img_file in os.listdir(image_dir)]
for img_path in tqdm(image_files, desc="Processing images"):
    img_data = preprocess_image(img_path)
    if img_data is not None:
        features = extract_features(img_data, model)
        if features is not None:
            features = features.flatten()  # Flatten the features to a 1D array
            data.append([os.path.basename(img_path)] + features.tolist())

# Create a DataFrame
columns = ['Image'] + [f'Feature_{i}' for i in range(len(features.flatten()))]
df = pd.DataFrame(data, columns=columns)

# Save to Excel
output_path = r'C:\Users\HP\Desktop\model\label_image_extraction.xlsx'
df.to_excel(output_path, index=False)

print("Feature extraction and saving to Excel completed successfully.")
