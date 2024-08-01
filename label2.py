import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.models import load_model
from tqdm import tqdm

# Load the trained model
model = load_model(r'C:\Users\HP\Desktop\model\trained_model.h5')

# Define class labels (update based on your training labels)
class_labels = {
    0: 'boys shoes',
    1: 'boys slippers',
    2: 'girls slippers',
    3: 'tshirt',
    4: 'shorts',
    5: 'girls vest',
    6: 'shirt',
    7: 'girls shoes',
    8: 'girls heels',
    9: 'boys formal shoes'
}

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return img_data

# Directory containing new images
image_dir = r'C:\Users\HP\Desktop\model\Dataset'
image_files = [os.path.join(image_dir, img_file) for img_file in os.listdir(image_dir)]

# List to hold predictions
data = []

# Process each image in the directory
for img_path in tqdm(image_files, desc="Labeling images"):
    img_data = preprocess_image(img_path)
    prediction = model.predict(img_data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_label = class_labels.get(predicted_class, 'Unknown')
    data.append([os.path.basename(img_path), class_label])

# Create a DataFrame and save to Excel
df = pd.DataFrame(data, columns=['Image', 'Label'])
df.to_excel(r'C:\Users\HP\Desktop\model\labeled_images.xlsx', index=False)

print("Labeling and saving to Excel completed successfully.")
