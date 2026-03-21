import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Setup Python path for imports
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(THIS_DIR)

# Define paths - single source of truth
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# Load models
if not os.path.exists(MODEL_DIR):
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
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train(dataset_dir=DATASET_DIR, model_out=MODEL_DIR, epochs=15, batch_size=32):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.15,
                                       rotation_range=10, horizontal_flip=True)
    train_gen = train_datagen.flow_from_directory(dataset_dir,
                                                  target_size=(64,64),
                                                  batch_size=batch_size,
                                                  class_mode='binary', subset='training')
    val_gen = train_datagen.flow_from_directory(dataset_dir,
                                                target_size=(64,64),
                                                batch_size=batch_size,
                                                class_mode='binary', subset='validation')
    model = build_model((64,64,3))
    cb = [tf.keras.callbacks.ModelCheckpoint(model_out, save_best_only=True), tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    hist = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=cb)
    # save plots
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    plt.figure()
    plt.plot(hist.history['accuracy'], label='acc')
    plt.plot(hist.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'accuracy_plot.png'))
    plt.figure()
    plt.plot(hist.history['loss'], label='loss')
    plt.plot(hist.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'loss_plot.png'))
    print('Training complete. Model saved to', model_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=DATASET_DIR)
    parser.add_argument('--out', default=MODEL_DIR)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    train(dataset_dir=args.dataset, model_out=args.out, epochs=args.epochs, batch_size=args.batch_size)

