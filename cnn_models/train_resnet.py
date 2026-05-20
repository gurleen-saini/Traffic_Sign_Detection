import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras import layers, models

from data_loader import load_data

DATA_DIR = "cnn_test_small"

# -----------------------------------
# LOAD DATASET
# -----------------------------------

train_ds, val_ds = load_data(DATA_DIR)

# -----------------------------------
# APPLY RESNET PREPROCESSING
# -----------------------------------

train_ds = train_ds.map(
    lambda x, y: (preprocess_input(x), y)
)

val_ds = val_ds.map(
    lambda x, y: (preprocess_input(x), y)
)

# -----------------------------------
# PREFETCH OPTIMIZATION
# -----------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# -----------------------------------
# LOAD PRETRAINED RESNET50
# -----------------------------------

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# -----------------------------------
# FREEZE MOST LAYERS
# -----------------------------------

for layer in base_model.layers[:-50]:
    layer.trainable = False

# -----------------------------------
# CUSTOM CLASSIFIER
# -----------------------------------

x = base_model.output

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(128, activation='relu')(x)

output = layers.Dense(43, activation='softmax')(x)

# -----------------------------------
# FINAL MODEL
# -----------------------------------

model = models.Model(
    inputs=base_model.input,
    outputs=output
)

# -----------------------------------
# COMPILE
# -----------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------------
# TRAIN
# -----------------------------------

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# -----------------------------------
# SAVE MODEL
# -----------------------------------

model.save("models/resnet.h5")

print("\n✅ ResNet model saved")