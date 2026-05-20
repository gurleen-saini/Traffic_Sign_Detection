import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras import layers, models

from data_loader import load_data

DATA_DIR = "cnn_test_small"

# Load dataset
train_ds, val_ds = load_data(DATA_DIR)

# Apply preprocessing
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

# Optimize loading
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Load pretrained MobileNet
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze most layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Custom classifier
x = base_model.output

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(128, activation='relu')(x)

output = layers.Dense(43, activation='softmax')(x)

# Final model
model = models.Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Save model
model.save("models/mobilenet.h5")

print("✅ MobileNet model saved")