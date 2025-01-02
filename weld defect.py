import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Image Preprocessing
img_size = (128, 128)  # Resize images to 128x128
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2  # 20% of data for validation
)

train_generator = datagen.flow_from_directory(
    'weld_images/',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training

',
)

validation_generator = datagen.flow_from_directory(
    'weld_images/',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
)

# Step 2: Build a CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 3: Train the Model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# Step 4: Defect Dimension Analysis
def analyze_defect(image_path):
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to isolate defect
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Bounding box for the defect
        x, y, w, h = cv2.boundingRect(contour)
        defect_dimension = (w, h)  # Width and height of the defect
        
        # Draw rectangle on the original image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        print(f"Defect Dimensions (Width x Height): {w} x {h}")
    
    # Display the result
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Defect Dimensions")
    plt.show()

# Example usage of defect analysis
test_image_path = 'test_weld_image.jpg'
analyze_defect(test_image_path)