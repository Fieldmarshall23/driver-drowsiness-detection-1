import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

DATA_DIR = 'dataset'
MODEL_DIR = 'models'
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 20

os.makedirs(MODEL_DIR, exist_ok=True)

def build_model(input_shape=(64,64,1)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("🚀 Starting Mouth CNN Training...")
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"📁 Model directory: {MODEL_DIR}")

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.7,1.3),
        zoom_range=0.1
    )

    print("🔄 Loading training data...")
    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        classes=['no_yawn', 'yawn'],
        class_mode='binary',
        batch_size=BATCH_SIZE,
        subset='training'
    )

    print("🔄 Loading validation data...")
    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        classes=['no_yawn', 'yawn'],
        class_mode='binary',
        batch_size=BATCH_SIZE,
        subset='validation'
    )

    print("🏗️ Building model...")
    model = build_model((IMG_SIZE[0], IMG_SIZE[1], 1))
    print("✅ Model built successfully")

    print(f"🎯 Starting training for {EPOCHS} epochs...")
    history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)

    print("💾 Saving model...")
    model.save(os.path.join(MODEL_DIR, 'mouth_cnn_model.h5'))

    print("✔ Mouth CNN model saved as mouth_cnn_model.h5")
    print("🎉 Training completed successfully!")

if __name__ == '__main__':
    main()

