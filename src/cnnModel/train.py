import os
import sys
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split


# Add current directory to path to allow imports
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from dataLoader import loadALSData
from model import createHybridModel

# Configuration
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
DATA_DIR = "../../Data/processed" # Relative to this script

def main():
    # 1. Load Data
    print("Loading data...")
    # Get absolute path to data
    scriptDir = Path(__file__).resolve().parent
    absDataDir = scriptDir / DATA_DIR
    
    if not absDataDir.exists():
        print(f"Error: Data directory not found at {absDataDir}")
        return

    X, y = loadALSData(absDataDir)
    
    if len(X) == 0:
        print("No data loaded. Please run preprocessing first.")
        return

    # 2. Split Data (80% Train, 20% Test)
    # Note: In medical imaging, split by Subject ID (which we did implicitly by loading subject-wise)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # 3. Create Model
    print("Creating Hybrid RViT Model...")
    model = createHybridModel()
    model.summary()

    # 4. Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy', # for one-hot encoded labels
        metrics=['accuracy']
    )

    # 5. Train
    print("Starting training...")
    # We use a checkpoint to save the best model
    checkpointPath = scriptDir / "best_model.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpointPath),
            save_best_only=True,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    print("Training complete.")

    # 6. Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
