import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import time
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

BASE_DIR = Path(os.getcwd())
# CSV files with dataset splits
TRAIN_CSV = 'train_augmented_fixed.csv'
VAL_CSV = 'splits/val_fixed.csv'
TEST_CSV = 'splits/test_fixed.csv'

BATCH_SIZE = 32
IMG_SIZE = 224  
EPOCHS = 10
LEARNING_RATE = 1e-4

CLASS_NAMES = ['clear', 'rain', 'fog', 'snow']
NUM_CLASSES = len(CLASS_NAMES)

RESNET_URL = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"

def load_dataset_from_csv(csv_path, is_training=False):
    """
    Load dataset from CSV file containing paths and labels
    
    Args:
        csv_path: Path to the CSV file
        is_training: Whether this is the training set (for augmentation)
        
    Returns:
        DataFrame with absolute filepaths and labels
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} images from {csv_path}")
    
    # converting relative paths to absolute paths
    df['filepath'] = df['filepath'].apply(lambda x: str(BASE_DIR / x))
    
    # converting numeric labels to string class names
    df['label'] = df['label'].apply(lambda x: CLASS_NAMES[int(x)])
    
    return df

def save_training_plots(history, model_name):
    """Save accuracy and loss plots from training history"""
    plots_dir = BASE_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # plotting accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plots_dir / f'{model_name}_training_history.png')
    plt.close()
    
    print(f"Training plots saved to {plots_dir / f'{model_name}_training_history.png'}")

def train_evaluate_resnet():
    """Train and evaluate ResNet model"""
    print("\n==================================================")
    print("TRAINING RESNET MODEL")
    print("==================================================")
    
    # loading datasets
    train_df = load_dataset_from_csv(TRAIN_CSV, is_training=True)
    val_df = load_dataset_from_csv(VAL_CSV)
    test_df = load_dataset_from_csv(TEST_CSV)
    
    # creating data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.0
    )
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=True
    )
    
    val_generator = train_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=False
    )
    
    test_generator = train_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=False
    )
    
    print("Building ResNet model...")
    
    # creating the base model using a Lambda wrapper to avoid symbolic tensor issues
    def hub_layer(x):
        return hub.KerasLayer(RESNET_URL, trainable=False)(x)

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.Lambda(hub_layer, output_shape=(2048,))(inputs)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)

    resnet_model = tf.keras.Model(inputs, outputs)
    
    # compiling model
    resnet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # creating callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_resnet_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3
        )
    ]
    
    # training model
    history = resnet_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # evaluating model
    test_loss, test_accuracy = resnet_model.evaluate(test_generator)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # getting predictions for confusion matrix
    print("\nGenerating confusion matrix...")
    predictions = resnet_model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # creating confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # --- Additional evaluation metrics ---
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy_score(true_classes, predicted_classes):.4f}")
    print(f"Sensitivity (Recall, macro): {recall_score(true_classes, predicted_classes, average='macro'):.4f}")
    print(f"Sensitivity (Recall, micro): {recall_score(true_classes, predicted_classes, average='micro'):.4f}")
    print(f"F1 Score (macro): {f1_score(true_classes, predicted_classes, average='macro'):.4f}")
    print(f"F1 Score (micro): {f1_score(true_classes, predicted_classes, average='micro'):.4f}")
    print(f"Precision (macro): {precision_score(true_classes, predicted_classes, average='macro'):.4f}")
    print(f"Precision (micro): {precision_score(true_classes, predicted_classes, average='micro'):.4f}")
    
    # creating plots directory if it doesn't exist
    plots_dir = BASE_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # plotting confusion matrix
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(
        cmap='YlOrRd',  # Warmer colors for better visibility
        values_format='d',  # Integer format for counts
        include_values=True,
        xticks_rotation=45,  # Rotate labels for better readability
        colorbar=True,
        text_kw={'size': 12},  # Larger text size
        im_kw={'vmin': 0}  # Start color scale from 0
    )
    plt.title('ResNet Confusion Matrix', pad=20, size=14, weight='bold')
    plt.ylabel('True Label', size=12, weight='bold')
    plt.xlabel('Predicted Label', size=12, weight='bold')
    plt.tight_layout()
    
    # saving confusion matrix plot
    plt.savefig(plots_dir / 'resnet_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {plots_dir / 'resnet_confusion_matrix.png'}")
    
    # saving training plots
    save_training_plots(history, 'resnet')
    
    return resnet_model

def create_vision_transformer(image_size, patch_size, num_patches, projection_dim, num_heads, transformer_units, transformer_layers, mlp_head_units):
    """Create a Vision Transformer model"""
    
    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = tf.keras.layers.Dense(units, activation='gelu')(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        return x
    
    class Patches(tf.keras.layers.Layer):
        def __init__(self, patch_size):
            super(Patches, self).__init__()
            self.patch_size = patch_size
        
        def call(self, images):
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            return patches
    
    class PatchEncoder(tf.keras.layers.Layer):
        def __init__(self, num_patches, projection_dim):
            super(PatchEncoder, self).__init__()
            self.num_patches = num_patches
            self.projection = tf.keras.layers.Dense(units=projection_dim)
            self.position_embedding = tf.keras.layers.Embedding(
                input_dim=num_patches, output_dim=projection_dim
            )
        
        def call(self, patch):
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            encoded = self.projection(patch) + self.position_embedding(positions)
            return encoded
    
    inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    
    # Create patches
    patches = Patches(patch_size)(inputs)
    
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    # Create multiple layers of the Transformer block
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Create a multi-head attention layer
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        
        # Skip connection 1
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        
        # Layer normalization 2
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        
        # Skip connection 2
        encoded_patches = tf.keras.layers.Add()([x3, x2])
    
    # Create a [batch_size, projection_dim] tensor
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = tf.keras.layers.GlobalAveragePooling1D()(representation)
    representation = tf.keras.layers.Dropout(0.5)(representation)
    
    # Add MLP
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    
    # Classify outputs
    logits = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(features)
    
    # Create the Keras model
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model

def train_evaluate_vit():
    """Train and evaluate Vision Transformer model using custom implementation"""
    print("\n==================================================")
    print("TRAINING VISION TRANSFORMER MODEL")
    print("==================================================")
    
    # ViT hyperparameters
    image_size = IMG_SIZE
    patch_size = 16
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]
    transformer_layers = 8
    mlp_head_units = [2048, 1024]
    
    # loading datasets
    train_df = load_dataset_from_csv(TRAIN_CSV, is_training=True)
    val_df = load_dataset_from_csv(VAL_CSV)
    test_df = load_dataset_from_csv(TEST_CSV)
    
    # creating data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.0
    )
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=True
    )
    
    val_generator = train_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=False
    )
    
    test_generator = train_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=False
    )
    
    print("Building Vision Transformer model...")
    
    # creating the custom ViT model
    vit_model = create_vision_transformer(
        image_size=image_size,
        patch_size=patch_size,
        num_patches=num_patches,
        projection_dim=projection_dim,
        num_heads=num_heads,
        transformer_units=transformer_units,
        transformer_layers=transformer_layers,
        mlp_head_units=mlp_head_units
    )
    
    # compiling model with a lower learning rate for ViT
    vit_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE * 0.1),  # Lower LR for ViT
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # creating callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_vit_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,  # More patience for ViT
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=4
        )
    ]
    
    # training model
    print("Starting ViT training...")
    history = vit_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # evaluating model
    test_loss, test_accuracy = vit_model.evaluate(test_generator)
    print(f"\nViT Test accuracy: {test_accuracy:.4f}")
    
    # getting predictions for confusion matrix
    print("\nGenerating ViT confusion matrix...")
    predictions = vit_model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # creating confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy_score(true_classes, predicted_classes):.4f}")
    print(f"Sensitivity (Recall, macro): {recall_score(true_classes, predicted_classes, average='macro'):.4f}")
    print(f"Sensitivity (Recall, micro): {recall_score(true_classes, predicted_classes, average='micro'):.4f}")
    print(f"F1 Score (macro): {f1_score(true_classes, predicted_classes, average='macro'):.4f}")
    print(f"F1 Score (micro): {f1_score(true_classes, predicted_classes, average='micro'):.4f}")
    print(f"Precision (macro): {precision_score(true_classes, predicted_classes, average='macro'):.4f}")
    print(f"Precision (micro): {precision_score(true_classes, predicted_classes, average='micro'):.4f}")

    # creating plots directory if it doesn't exist
    plots_dir = BASE_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # plotting confusion matrix
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(
        cmap='Blues',  
        values_format='d',
        include_values=True,
        xticks_rotation=45,
        colorbar=True,
        text_kw={'size': 12},
        im_kw={'vmin': 0}
    )
    plt.title('Vision Transformer Confusion Matrix', pad=20, size=14, weight='bold')
    plt.ylabel('True Label', size=12, weight='bold')
    plt.xlabel('Predicted Label', size=12, weight='bold')
    plt.tight_layout()
    
    # saving confusion matrix plot
    plt.savefig(plots_dir / 'vit_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ViT Confusion matrix saved to {plots_dir / 'vit_confusion_matrix.png'}")
    
    # saving training plots
    save_training_plots(history, 'ViT')
    
    return vit_model


if __name__ == "__main__":
    print("Weather Classification Model Training")
    print("=" * 40)
    print(f"Class mapping: {CLASS_NAMES}")
    print(f"CSV files: {TRAIN_CSV}, {VAL_CSV}, {TEST_CSV}")
    
    # choosing which model to train
    train_resnet = True    # Set to False to skip ResNet training
    train_vit = True      # Set to False to skip ViT training
    
    # training models
    if train_resnet:
        resnet_model = train_evaluate_resnet()
    
    if train_vit:
        vit_model = train_evaluate_vit()
    
    print("\nTraining complete!")
    print("Models are saved in the current directory and 'models' directory")
    print("Training plots are saved in the 'plots' directory")