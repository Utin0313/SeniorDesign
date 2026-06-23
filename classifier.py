import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# =========================
# CONFIG
# =========================
INPUT_DIR = "Data_new_final2"
BATCH_SIZE = 32
EPOCHS = 20
IMG_SIZE = (224, 224)
LR = 1e-5
MODEL_SAVE_PATH = "resnet50_simple_classifier.keras"

# =========================
# DATA AUGMENTATION
# =========================
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    rotation_range=5,
    horizontal_flip=True,      
    vertical_flip=False,       
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode="nearest"
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

# =========================
# LOAD DATA
# =========================
train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(INPUT_DIR, "Train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    directory=os.path.join(INPUT_DIR, "Validation"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    directory=os.path.join(INPUT_DIR, "Test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# =========================
# CHECK CLASS ORDER (VERY IMPORTANT)
# =========================
print("Class Indices:", train_generator.class_indices)

CLASS_NAMES = [k for k, v in sorted(train_generator.class_indices.items(), key=lambda x: x[1])]
NUM_CLASSES = len(CLASS_NAMES)

print("Final Class Order:", CLASS_NAMES)

# =========================
# CLASS WEIGHTS (FIXED)
# =========================
labels = train_generator.classes

class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = {
    cls: weight
    for cls, weight in zip(np.unique(labels), class_weights_arr)
}

print("Class Weights:", class_weights)

# =========================
# MODEL
# =========================
base_model = ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# freeze base first
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)

# classifier head
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# =========================
# FINE TUNING SETUP
# =========================
base_model.trainable = True

# freeze early layers, train last ~30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# =========================
# COMPILE
# =========================
model.compile(
    optimizer=optimizers.Adam(learning_rate=LR, clipnorm=1.0),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# CALLBACKS
# =========================
cb = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    ),
    callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# =========================
# TRAIN
# =========================
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=cb
)

# =========================
# PLOTS
# =========================
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")
plt.savefig("loss.png")
plt.close()

plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Accuracy")
plt.savefig("accuracy.png")
plt.close()

print("Training complete. Model saved to:", MODEL_SAVE_PATH)
