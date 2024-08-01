import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os

# Load the labels from the Excel file
excel_file = r'C:\Users\HP\Desktop\model\extracted_features_with_labels.xlsx'
df = pd.read_excel(excel_file)

# Prepare the image filenames and labels
image_paths = df['Image'].values
labels = df['Label'].values

# Convert labels to categorical format
label_dict = {label: idx for idx, label in enumerate(np.unique(labels))}
num_classes = len(label_dict)
labels = np.array([label_dict[label] for label in labels])
labels = to_categorical(labels, num_classes=num_classes)

# Define a function to preprocess the images
def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Create dataset
def create_dataset(image_paths, labels):
    images = []
    for path in image_paths:
        img = load_and_preprocess_image(path)
        images.append(img)
    return np.vstack(images), labels

# Create dataset
image_dir = r'C:\Users\HP\Desktop\model\Dataset'
image_paths = [os.path.join(image_dir, fname) for fname in image_paths]
images, labels = create_dataset(image_paths, labels)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Load the base ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

# Save the trained model
model.save(r'C:\Users\HP\Desktop\model\trained_model.h5')

print("Model training completed and saved successfully.")
import json

# After training the model
class_labels = {v: k for k, v in train_generator.class_indices.items()}

# Save class labels to a JSON file
with open(r'C:\Users\HP\Desktop\model\class_labels.json', 'w') as f:
    json.dump(class_labels, f)
