import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score,
    classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
import tensorflow_hub as hub
import csv

# --- Config ---
BASE_DIR = Path(os.getcwd())
TEST_CSV = 'splits/test_fixed.csv'
MODEL_PATH = 'best_resnet_model.h5'
CLASS_NAMES = ['clear', 'rain', 'fog', 'snow']
IMG_SIZE = 224
BATCH_SIZE = 32

# --- Define hub_layer for custom_objects ---
def hub_layer(x):
    return hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5", trainable=False)(x)

# --- Load test data ---
test_df = pd.read_csv(TEST_CSV)
test_df['filepath'] = test_df['filepath'].apply(lambda x: str(BASE_DIR / x))
test_df['label'] = test_df['label'].apply(lambda x: CLASS_NAMES[int(x)])

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="filepath",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=False
)

# --- Load model with custom_objects ---
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'hub_layer': hub_layer}
)

# --- Predict ---
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# --- Metrics ---
acc = accuracy_score(true_classes, predicted_classes)
recall_macro = recall_score(true_classes, predicted_classes, average='macro')
f1_macro = f1_score(true_classes, predicted_classes, average='macro')
precision_macro = precision_score(true_classes, predicted_classes, average='macro')
class_report = classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES, digits=4, output_dict=True)

print("\nEvaluation Metrics:")
print(f"Accuracy: {acc:.4f}")
print(f"Recall (macro): {recall_macro:.4f}")
print(f"F1 Score (macro): {f1_macro:.4f}")
print(f"Precision (macro): {precision_macro:.4f}")

print("\nDetailed Classification Report (per class):")
print(classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES, digits=4))

# --- Save metrics to CSV ---
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