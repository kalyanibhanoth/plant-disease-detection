import tensorflow as tf
from tensorflow.keras import layers, models
import json
import os

# ---------------------------------------
# Settings
# ---------------------------------------

DATASET_PATH = "dataset"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3

# ---------------------------------------
# Load dataset
# ---------------------------------------

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# ---------------------------------------
# Get disease names
# ---------------------------------------

class_names = train_dataset.class_names

print("Disease classes:")
for i, name in enumerate(class_names):
    print(i, ":", name)

# Save disease names
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# ---------------------------------------
# Improve performance
# ---------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

# ---------------------------------------
# Image preprocessing
# ---------------------------------------

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# ---------------------------------------
# CNN Model
# ---------------------------------------

model = models.Sequential([

    layers.Input(shape=(224, 224, 3)),

    # Image preprocessing
    layers.Rescaling(1.0 / 255),

    data_augmentation,

    # Convolution 1
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    # Convolution 2
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    # Convolution 3
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    # Convolution 4
    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    # Classification
    layers.Flatten(),

    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(
        len(class_names),
        activation="softmax"
    )
])

# ---------------------------------------
# Compile
# ---------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------------------
# Train
# ---------------------------------------

print("\nStarting training...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# ---------------------------------------
# Save model
# ---------------------------------------

model.save("plant_disease_model.keras")

print("\n--------------------------------")
print("Training completed!")
print("--------------------------------")
print("Model saved as:")
print("plant_disease_model.keras")

print("\nClass names saved as:")
print("class_names.json")