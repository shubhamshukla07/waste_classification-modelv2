import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import os

# Set your dataset path here
dataset_path = r'C:\Users\shubh\OneDrive\Desktop\image_classification\waste_classification\new-dataset-trash-type-v2'

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

# Load training dataset with validation split
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

# Preprocess input for ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

# Optimize performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Load pre-trained ResNet50 base
base_model = tf.keras.applications.ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Define number of classes
num_classes = 9

# Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Initial training
print("Starting initial training with frozen base model...")
history_initial = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop, lr_reduce]
)

# Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Starting fine-tuning of last 30 layers...")

# ModelCheckpoint callback to save best model during fine-tuning
checkpoint = callbacks.ModelCheckpoint(
    filepath=r'C:\Users\shubh\OneDrive\Desktop\image_classification\best_model_tf.keras',
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[early_stop, lr_reduce, checkpoint]
)

# Save final model after training finishes (optional)
save_path = r'C:\Users\shubh\OneDrive\Desktop\image_classification\final_model_tf'
model.save(save_path)
print(f"✅ Final model saved to: {save_path}")

