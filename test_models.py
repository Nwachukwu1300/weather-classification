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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

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
    x = tf.keras.layers.Lambda(hub_layer)(inputs)
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

def train_evaluate_vit():
    """Train and evaluate Vision Transformer model using HuggingFace"""
    from transformers import ViTFeatureExtractor, ViTForImageClassification
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    
    print("\n" + "="*50)
    print("TRAINING VISION TRANSFORMER MODEL")
    print("="*50)
    
    # checking for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # creating directory for models if it doesn't exist
    models_dir = BASE_DIR / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # creating custom dataset class
    class WeatherDataset(Dataset):
        def __init__(self, csv_file, feature_extractor):
            self.df = pd.read_csv(csv_file)
            # converting relative paths to absolute
            self.df['filepath'] = self.df['filepath'].apply(lambda x: str(BASE_DIR / x))
            self.feature_extractor = feature_extractor
            
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            # getting image path and label
            img_path = self.df.iloc[idx]['filepath']
            label = self.df.iloc[idx]['label']
            
            # opening image
            image = Image.open(img_path).convert('RGB')
            
            # processing image
            encoding = self.feature_extractor(image, return_tensors='pt')
            # removing batch dimension
            for k, v in encoding.items():
                encoding[k] = v.squeeze()
                
            return {
                'pixel_values': encoding['pixel_values'],
                'labels': torch.tensor(label, dtype=torch.long)
            }
    
    # loading feature extractor
    print("Loading ViT feature extractor...")
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        'google/vit-base-patch16-224-in21k'
    )
    
    # loading model
    print("Loading ViT model...")
    vit_model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=NUM_CLASSES,
        id2label={i: name for i, name in enumerate(CLASS_NAMES)},
        label2id={name: i for i, name in enumerate(CLASS_NAMES)}
    )
    vit_model = vit_model.to(device)
    
    # creating datasets
    print("Creating datasets...")
    train_dataset = WeatherDataset(TRAIN_CSV, feature_extractor)
    val_dataset = WeatherDataset(VAL_CSV, feature_extractor)
    test_dataset = WeatherDataset(TEST_CSV, feature_extractor)
    
    # creating data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    print(f"Training on {len(train_dataset)} samples")
    print(f"Validating on {len(val_dataset)} samples")
    print(f"Testing on {len(test_dataset)} samples")
    
    # setting up optimizer
    optimizer = torch.optim.AdamW(vit_model.parameters(), lr=5e-5)
    
    # training loop
    print(f"Training ViT model for {EPOCHS} epochs...")
    best_val_accuracy = 0.0
    history = {
        'accuracy': [],
        'val_accuracy': [],
        'loss': [],
        'val_loss': []
    }
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        # training phase
        vit_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # zero gradients
            optimizer.zero_grad()
            
            # forward pass
            outputs = vit_model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            # backward pass
            loss.backward()
            optimizer.step()
            
            # tracking statistics
            train_loss += loss.item()
            _, predicted = outputs.logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Calculate training metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = train_correct / train_total
        
        # validation phase
        vit_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = vit_model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                
                val_loss += loss.item()
                _, predicted = outputs.logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # calculating validation metrics
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = val_correct / val_total
        
        # updating history
        history['accuracy'].append(epoch_train_acc)
        history['val_accuracy'].append(epoch_val_acc)
        history['loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        # saving best model
        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            torch.save(vit_model.state_dict(), models_dir / 'vit_best.pt')
            print(f"Saved best model with validation accuracy: {best_val_accuracy:.4f}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # saving final model
    torch.save(vit_model.state_dict(), models_dir / 'vit_final.pt')
    print(f"Final model saved to {models_dir / 'vit_final.pt'}")
    
    # loading best model for evaluation
    vit_model.load_state_dict(torch.load(models_dir / 'vit_best.pt'))
    
    # evaluating on test set
    print("Evaluating on test set...")
    vit_model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = vit_model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            test_loss += loss.item()
            _, predicted = outputs.logits.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_accuracy = test_correct / test_total
    test_loss = test_loss / len(test_loader)
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # saving training plots
    save_training_plots(history, 'ViT')
    
    # returning the model and feature extractor for later use
    return vit_model, feature_extractor


if __name__ == "__main__":
    print("Weather Classification Model Training")
    print("=" * 40)
    print(f"Class mapping: {CLASS_NAMES}")
    print(f"CSV files: {TRAIN_CSV}, {VAL_CSV}, {TEST_CSV}")
    
    # choosing which model to train
    train_resnet = True  # Set to False to skip ResNet training
    train_vit = False    # Set to False to skip ViT training
    
    # training models
    if train_resnet:
        resnet_model = train_evaluate_resnet()
    
    if train_vit:
        vit_model, feature_extractor = train_evaluate_vit()
    
    print("\nTraining complete!")
    print("Models are saved in the 'models' directory")
    print("Training plots are saved in the 'plots' directory")