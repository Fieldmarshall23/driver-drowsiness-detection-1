# ========================================
# EYE CNN TRAINING SCRIPT
# ========================================
# Trains binary CNN classifier for eye open/closed detection
# Dataset: dataset/open_eyes/ vs dataset/closed_eyes/
# Output: models/cnn_model.h5

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ========================================
# PATH SETUP
# ========================================
# Resolve project paths relative to script location
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(THIS_DIR)

# Standard paths for data/models/outputs
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# Ensure model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)


# ========================================
# MODEL ARCHITECTURE
# ========================================
# Simple CNN: 3 Conv+Pool -> Flatten -> Dense -> Sigmoid
# Input: (64,64,3) grayscale eyes repeated to RGB
def build_model(input_shape=(64,64,3)):  # Note: RGB (grayscale repeated)
    """
    Build 3-layer CNN for binary eye classification (open=0, closed=1).
    
    Architecture:
    - Conv32(3x3)-Pool -> Conv64(3x3)-Pool -> Conv128(3x3)-Pool
    - Flatten -> Dense128(ReLU) -> Dense1(Sigmoid)
    
    Args:
        input_shape: Expected (64,64,3)
    
    Returns:
        Compiled Keras Sequential model
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model


# ========================================
# TRAINING PIPELINE
# ========================================
# Loads data via ImageDataGenerator (aug + split), trains with callbacks,
# saves best model + acc/loss plots
def train(dataset_dir=DATASET_DIR, model_out=os.path.join(MODEL_DIR, 'cnn_model.h5'), epochs=15, batch_size=32):
    """
    Train eye CNN model.
    
    Data: Assumes dataset_dir/open_eyes/ (0), dataset_dir/closed_eyes/ (1)
    Augmentation: rescale, 15% val split, rot10, hflip
    Callbacks: Best model checkpoint + early stop (patience=5)
    
    Args:
        dataset_dir: Path to dataset/
        model_out: Save path for cnn_model.h5
        epochs: Max training epochs
        batch_size: Batch size
    
    Saves:
        cnn_model.h5 (best val)
        outputs/accuracy_plot.png, loss_plot.png
    """
    # Data generator with augmentation and val split
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, 
        validation_split=0.15,
        rotation_range=10, 
        horizontal_flip=True
    )
    
    # Training generator
    train_gen = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(64,64),
        batch_size=batch_size,
        class_mode='binary', 
        subset='training'
    )
    
    # Validation generator
    val_gen = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(64,64),
        batch_size=batch_size,
        class_mode='binary', 
        subset='validation'
    )
    
    # Build and train model
    model = build_model((64,64,3))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_out, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
    hist = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    
    # ========================================
    # SAVE TRAINING PLOTS
    # ========================================
    # Visualize acc/loss curves
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    
    # Accuracy plot
    plt.figure()
    plt.plot(hist.history['accuracy'], label='train_acc')
    plt.plot(hist.history['val_accuracy'], label='val_acc')
    plt.title('Eye CNN Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'eye_accuracy_plot.png'))
    plt.close()
    
    # Loss plot
    plt.figure()
    plt.plot(hist.history['loss'], label='train_loss')
    plt.plot(hist.history['val_loss'], label='val_loss')
    plt.title('Eye CNN Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'eye_loss_plot.png'))
    plt.close()
    
    print(f'Training complete. Model saved to {model_out}')


# ========================================
# CLI ENTRYPOINT
# ========================================
# Usage: python src/eye_cnn_training.py --dataset dataset/ --epochs 20
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Eye CNN for open/closed classification')
    parser.add_argument('--dataset', default=DATASET_DIR, help='Path to dataset/')
    parser.add_argument('--out', default=os.path.join(MODEL_DIR, 'cnn_model.h5'), help='Output model path')
    parser.add_argument('--epochs', type=int, default=15, help='Max epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    args = parser.parse_args()
    train(dataset_dir=args.dataset, model_out=args.out, epochs=args.epochs, batch_size=args.batch_size)


