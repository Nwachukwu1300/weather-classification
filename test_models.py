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
from sklearn.metrics import classification_report
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from torch.utils.data import Dataset, DataLoader
import optuna
from optuna.trial import Trial
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

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

def objective(trial: Trial):
    """Optuna objective function for hyperparameter optimization"""
    # defining hyperparameters to optimize
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.7)
    dense_units = trial.suggest_int('dense_units', 256, 1024)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    num_dense_layers = trial.suggest_int('num_dense_layers', 1, 3)
    dense_units_multiplier = trial.suggest_float('dense_units_multiplier', 0.5, 1.0)
    
    # loading datasets
    train_df = load_dataset_from_csv(TRAIN_CSV, is_training=True)
    val_df = load_dataset_from_csv(VAL_CSV)
    
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
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=True
    )
    
    val_generator = train_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False
    )
    
    # creating the base model
    def hub_layer(x):
        return hub.KerasLayer(RESNET_URL, trainable=False)(x)

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.Lambda(hub_layer, output_shape=(2048,))(inputs)
    
    # Add multiple dense layers with decreasing units
    current_units = dense_units
    for i in range(num_dense_layers):
        x = tf.keras.layers.Dense(current_units, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        current_units = int(current_units * dense_units_multiplier)
    
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)

    resnet_model = tf.keras.Model(inputs, outputs)
    
    # compiling model
    resnet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # creating callbacks with more patient early stopping for better accuracy
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,  # Increased patience for better convergence
            restore_best_weights=True,
            min_delta=0.0005  # Smaller delta for more precise stopping
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,  # Increased patience for learning rate reduction
            min_delta=0.0005
        )
    ]
    
    # training model with more epochs for optimization
    history = resnet_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,  # Increased epochs for better convergence
        callbacks=callbacks,
        verbose=0
    )
    
    # Return validation accuracy as the objective value
    return max(history.history['val_accuracy'])

def train_evaluate_resnet():
    """Train and evaluate ResNet model with best hyperparameters (Optuna code commented out)"""
    print("\n==================================================")
    print("TRAINING RESNET MODEL WITH BEST HYPERPARAMETERS (Optuna code commented out)")
    print("==================================================")
    
    # --- Commented out Optuna optimization ---
    # study = optuna.create_study(
    #     direction='maximize',
    #     pruner=optuna.pruners.MedianPruner(
    #         n_startup_trials=5,  # More startup trials
    #         n_warmup_steps=3,    # More warmup steps
    #         interval_steps=1
    #     )
    # )
    # study.optimize(
    #     objective,
    #     n_trials=10,
    #     timeout=72000,  # Increased to 2 hours
    #     show_progress_bar=True
    # )
    # print("\nBest trial:")
    # trial = study.best_trial
    # print("  Value: ", trial.value)
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print(f"    {key}: {value}")
    # print("\nTraining final model with best parameters...")

    # --- Use best hyperparameters found previously ---
    best_params = {
        'learning_rate': 0.000912,  # Example best value
        'dropout_rate': 0.3591,
        'dense_units': 650,
        'batch_size': 64,
        'num_dense_layers': 1,
        'dense_units_multiplier': 0.532
    }
    
    # loading datasets
    train_df = load_dataset_from_csv(TRAIN_CSV, is_training=True)
    val_df = load_dataset_from_csv(VAL_CSV)
    test_df = load_dataset_from_csv(TEST_CSV)
    
    # creating data generators with best batch size
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.0
    )
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=best_params['batch_size'],
        class_mode="sparse",
        shuffle=True
    )
    
    val_generator = train_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=best_params['batch_size'],
        class_mode="sparse",
        shuffle=False
    )
    
    test_generator = train_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=best_params['batch_size'],
        class_mode="sparse",
        shuffle=False
    )
    
    # creating the base model with best parameters
    def hub_layer(x):
        return hub.KerasLayer(RESNET_URL, trainable=False)(x)

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.Lambda(hub_layer, output_shape=(2048,))(inputs)
    
    # Add multiple dense layers with best parameters
    current_units = best_params['dense_units']
    for i in range(best_params['num_dense_layers']):
        x = tf.keras.layers.Dense(current_units, activation='relu')(x)
        x = tf.keras.layers.Dropout(best_params['dropout_rate'])(x)
        current_units = int(current_units * best_params['dense_units_multiplier'])
    
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)

    resnet_model = tf.keras.Model(inputs, outputs)
    
    # compiling model with best learning rate
    resnet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # creating callbacks for final training with more patient settings
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_resnet_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,  # Increased patience for final training
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=4,  # Increased patience for learning rate reduction
            min_delta=0.0005  # Smaller delta for more precise stopping
        )
    ]
    
    # training final model with more epochs
    history = resnet_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
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
    acc = accuracy_score(true_classes, predicted_classes)
    recall_macro = recall_score(true_classes, predicted_classes, average='macro')
    recall_micro = recall_score(true_classes, predicted_classes, average='micro')
    f1_macro = f1_score(true_classes, predicted_classes, average='macro')
    f1_micro = f1_score(true_classes, predicted_classes, average='micro')
    precision_macro = precision_score(true_classes, predicted_classes, average='macro')
    precision_micro = precision_score(true_classes, predicted_classes, average='micro')
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"Recall (micro): {recall_micro:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"F1 Score (micro): {f1_micro:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Precision (micro): {precision_micro:.4f}")
    
    # --- Per-class metrics ---
    print("\nDetailed Classification Report (per class):")
    class_report = classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES, digits=4, output_dict=True)
    print(classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES, digits=4))
    
    # Save evaluation metrics to CSV
    import csv
    plots_dir = BASE_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)
    metrics_csv = plots_dir / 'resnet_evaluation_metrics.csv'
    with open(metrics_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        for cls in CLASS_NAMES:
            row = class_report[cls]
            writer.writerow([cls, f"{row['precision']:.4f}", f"{row['recall']:.4f}", f"{row['f1-score']:.4f}", int(row['support'])])
        writer.writerow([])
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Accuracy', f"{acc:.4f}"])
        writer.writerow(['Macro Avg F1', f"{f1_macro:.4f}"])
        writer.writerow(['Weighted Avg F1', f"{class_report['weighted avg']['f1-score']:.4f}"])
    print(f"Evaluation metrics saved to {metrics_csv}")
    
    # --- ROC Curve (One-vs-Rest) ---
    print("\nPlotting ROC curve...")
    n_classes = len(CLASS_NAMES)
    true_classes_bin = label_binarize(true_classes, classes=list(range(n_classes)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_classes_bin[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green', 'orange']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {CLASS_NAMES[i]} (area = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ResNet Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = plots_dir / 'resnet_roc_curve.png'
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC curve saved to {roc_path}")
    
    # creating plots directory if it doesn't exist
    plots_dir = BASE_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # plotting confusion matrix
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(
        cmap='YlOrRd',
        values_format='d',
        include_values=True,
        xticks_rotation=45,
        colorbar=True,
        text_kw={'size': 12},
        im_kw={'vmin': 0}
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
    """Train and evaluate Vision Transformer model using Hugging Face google/vit-base-patch16-224"""
    print("\n==================================================")
    print("TRAINING VISION TRANSFORMER MODEL (Hugging Face)")
    print("==================================================")

    # Load datasets
    train_df = load_dataset_from_csv(TRAIN_CSV, is_training=True)
    val_df = load_dataset_from_csv(VAL_CSV)
    test_df = load_dataset_from_csv(TEST_CSV)

    # Prepare Hugging Face processor and model
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=NUM_CLASSES,
        id2label={i: name for i, name in enumerate(CLASS_NAMES)},
        label2id={name: i for i, name in enumerate(CLASS_NAMES)},
        ignore_mismatched_sizes=True  # Add this to handle size mismatch
    )
    # Configure the classification head for our 4 classes
    model.classifier = torch.nn.Linear(model.config.hidden_size, NUM_CLASSES)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Custom PyTorch dataset
    class WeatherDataset(Dataset):
        def __init__(self, df, processor):
            self.df = df.reset_index(drop=True)
            self.processor = processor
            # Convert string labels to numeric indices
            self.df['label'] = self.df['label'].apply(lambda x: CLASS_NAMES.index(x))
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            img_path = self.df.iloc[idx]['filepath']
            label = self.df.iloc[idx]['label']  # Now label is already numeric
            image = Image.open(img_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in inputs.items()}
            item['labels'] = torch.tensor(label, dtype=torch.long)
            return item

    train_dataset = WeatherDataset(train_df, processor)
    val_dataset = WeatherDataset(val_df, processor)
    test_dataset = WeatherDataset(test_df, processor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    EPOCHS_VIT = EPOCHS  # Use same EPOCHS as before
    best_val_acc = 0.0
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

    for epoch in range(EPOCHS_VIT):
        print(f"Epoch {epoch+1}/{EPOCHS_VIT}")
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = outputs.logits.argmax(dim=1)
            train_correct += (preds == batch['labels']).sum().item()
            train_total += batch['labels'].size(0)
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                val_loss += loss.item()
                preds = outputs.logits.argmax(dim=1)
                val_correct += (preds == batch['labels']).sum().item()
                val_total += batch['labels'].size(0)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = val_correct / val_total

        history['accuracy'].append(epoch_train_acc)
        history['val_accuracy'].append(epoch_val_acc)
        history['loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)

        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_vit_model.pt')
            print(f"Saved best model with validation accuracy: {best_val_acc:.4f}")

    # Load best model
    model.load_state_dict(torch.load('best_vit_model.pt'))
    model.eval()

    # Test evaluation
    test_loss, test_correct, test_total = 0.0, 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            test_loss += loss.item()
            preds = outputs.logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            test_correct += (preds == batch['labels']).sum().item()
            test_total += batch['labels'].size(0)
    test_acc = test_correct / test_total
    print(f"\nViT Test accuracy: {test_acc:.4f}")

    # Metrics
    print("\nGenerating ViT confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds)*100:.4f}")
    print(f"Sensitivity (Recall, macro): {recall_score(all_labels, all_preds, average='macro'):.4f}")
    print(f"Sensitivity (Recall, micro): {recall_score(all_labels, all_preds, average='micro'):.4f}")
    print(f"F1 Score (macro): {f1_score(all_labels, all_preds, average='macro'):.4f}")
    print(f"F1 Score (micro): {f1_score(all_labels, all_preds, average='micro'):.4f}")
    print(f"Precision (macro): {precision_score(all_labels, all_preds, average='macro'):.4f}")
    print(f"Precision (micro): {precision_score(all_labels, all_preds, average='micro'):.4f}")
    # --- Per-class metrics ---
    print("\nDetailed Classification Report (per class):")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

    # Plot confusion matrix
    plots_dir = BASE_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)
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
    plt.title('ViT Confusion Matrix (Hugging Face)', pad=20, size=14, weight='bold')
    plt.ylabel('True Label', size=12, weight='bold')
    plt.xlabel('Predicted Label', size=12, weight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / 'vit_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ViT Confusion matrix saved to {plots_dir / 'vit_confusion_matrix.png'}")

    # Save training plots (reuse existing function)
    save_training_plots(history, 'ViT')
    return model


if __name__ == "__main__":
    print("Weather Classification Model Training")
    print("=" * 40)
    print(f"Class mapping: {CLASS_NAMES}")
    print(f"CSV files: {TRAIN_CSV}, {VAL_CSV}, {TEST_CSV}")
    
    # choosing which model to train
    train_resnet =  True    # Set to False to skip ResNet training
    train_vit = False      # Set to False to skip ViT training
    
    # training models
    if train_resnet:
        resnet_model = train_evaluate_resnet()
    
    if train_vit:
        vit_model = train_evaluate_vit()
    
    print("\nTraining complete!")
    print("Models are saved in the current directory and 'models' directory")
    print("Training plots are saved in the 'plots' directory")