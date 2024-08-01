import numpy as np
import pandas as pd
import json
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Directory containing your images
image_dir = r'C:\Users\HP\Desktop\model\Dataset'

# Load the Excel file with image names, labels, and features
excel_path = r'C:\Users\HP\Desktop\model\extracted_features_with_labels.xlsx'
df = pd.read_excel(excel_path)

# Prepare image names, labels, and features
image_names = df['Image'].values
labels = df['Label'].values

# Define class labels and mapping
class_labels = sorted(set(labels))
class_indices = {label: index for index, label in enumerate(class_labels)}
num_classes = len(class_labels)

# Map labels to indices
labels_indices = [class_indices[label] for label in labels]

# Load and preprocess images
def load_and_preprocess_image(img_name):
    img_path = os.path.join(image_dir, img_name)  # Create the full path to the image
    if not os.path.isfile(img_path):
        print(f"File not found: {img_path}")
        return None
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

images = np.array([load_and_preprocess_image(name) for name in image_names if load_and_preprocess_image(name) is not None])
labels_one_hot = to_categorical(labels_indices, num_classes=num_classes)

# Create the model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    images,
    labels_one_hot,
    batch_size=32,
    epochs=10,
    validation_split=0.2
)

# Save the trained model
model.save(r'C:\Users\HP\Desktop\model\trained_model.h5')

# Save class labels
with open(r'C:\Users\HP\Desktop\model\class_labels.json', 'w') as f:
    json.dump(class_indices, f)

print("Training completed and model saved successfully.")
