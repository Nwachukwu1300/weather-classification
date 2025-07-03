# Weather Image Classification

A deep learning project for classifying weather conditions from images using ResNet and Vision Transformer (ViT) models.

## Overview

This project implements a weather classification system that can identify four different weather conditions:
- **Clear** - Clear skies
- **Rain** - Rainy conditions  
- **Fog** - Foggy weather
- **Snow** - Snowy conditions

The system uses state-of-the-art deep learning models including ResNet-50 and Vision Transformer (ViT) for accurate weather classification.

## Project Structure

```
.
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── weather_preprocessing.py       # Data preprocessing and splitting
├── augment_images.py             # Data augmentation script
├── evaluate_resnet.py            # ResNet model evaluation
├── test_models.py                # Comprehensive model testing
├── fix_paths.py                  # Path fixing utilities
├── show_distribution.py          # Dataset distribution visualization
├── best_resnet_model.h5          # Trained ResNet model
├── best_vit_model.h5            # Trained ViT model
├── train_augmented_fixed.csv     # Training data with augmentation
├── dataset_distribution.png      # Dataset distribution visualization
├── clear/                        # Clear weather images
├── rain/                         # Rain weather images
├── fog/                          # Fog weather images
├── snow/                         # Snow weather images
├── augmented_images/             # Data augmentation outputs
├── splits/                       # Train/validation/test splits
└── plots/                        # Evaluation plots and metrics
    ├── resnet_roc_curve.png
    ├── resnet_confusion_matrix.png
    ├── resnet_training_history.png
    ├── resnet_evaluation_metrics.csv
    ├── vit_confusion_matrix.png
    └── ViT_training_history.png
```

## Features

### 🔬 Data Processing
- **Automated dataset scanning** with image validation
- **Balanced dataset creation** with stratified sampling
- **Train/validation/test splits** (70%/15%/15%)
- **Data augmentation** for underrepresented classes
- **Class distribution visualization**

### 🤖 Model Architectures
- **ResNet-50** with TensorFlow Hub integration
- **Vision Transformer (ViT)** for state-of-the-art performance
- **Transfer learning** from pre-trained models
- **Custom training pipelines** for each architecture

### 📊 Evaluation & Metrics
- **Comprehensive evaluation metrics**: Accuracy, Precision, Recall, F1-Score
- **ROC curves** with AUC scores for each class
- **Confusion matrices** for detailed performance analysis
- **Training history visualization**
- **Per-class performance breakdown**

 **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Dependencies
- `tensorflow-macos` & `tensorflow-metal` (for Apple Silicon Macs)
- `tensorflow-hub` (for pre-trained models)
- `torch` & `torchvision` (PyTorch support)
- `transformers` (Hugging Face transformers)
- `scikit-learn` (evaluation metrics)
- `matplotlib` (visualizations)
- `pillow` (image processing)
- `pandas` (data manipulation)
- `optuna` (hyperparameter optimization)

## Usage

### 1. Data Preprocessing

Prepare your dataset and create balanced splits:

```bash
python weather_preprocessing.py
```

This script will:
- Scan all weather category folders for valid images
- Create balanced train/validation/test splits
- Generate dataset distribution visualization
- Save splits to CSV files in the `splits/` directory

### 2. Data Augmentation

Enhance your training data with augmentation:

```bash
python augment_images.py
```

Features include:
- Horizontal flipping
- Brightness adjustments
- Hue shifting
- Targeted augmentation for underrepresented classes

### 3. Model Evaluation

Evaluate the trained ResNet model:

```bash
python evaluate_resnet.py
```

Or run comprehensive testing on both models:

```bash
python test_models.py
```

### 4. Dataset Analysis

View dataset distribution:

```bash
python show_distribution.py
```

## Model Performance

### ResNet-50 Results
- Utilizes transfer learning from ImageNet
- Comprehensive evaluation with ROC curves
- Detailed per-class metrics available in `plots/resnet_evaluation_metrics.csv`

### Vision Transformer (ViT) Results
- State-of-the-art transformer architecture
- Competitive performance on weather classification
- Training history and confusion matrix visualizations available

## File Descriptions

| File | Purpose |
|------|---------|
| `weather_preprocessing.py` | Main preprocessing pipeline for dataset preparation |
| `augment_images.py` | Data augmentation for improving model robustness |
| `evaluate_resnet.py` | Comprehensive evaluation of ResNet model |
| `test_models.py` | Testing framework for multiple model architectures |
| `fix_paths.py` | Utility for fixing file paths in datasets |
| `show_distribution.py` | Visualization of class distribution |

## Results & Visualizations

The project generates comprehensive evaluation materials:

- **ROC Curves**: Multi-class ROC analysis for each weather condition
- **Confusion Matrices**: Detailed classification performance breakdown
- **Training History**: Loss and accuracy progression during training
- **Metrics CSV**: Exportable performance metrics for further analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is available for educational and research purposes.

## Acknowledgments

- TensorFlow Hub for pre-trained ResNet models
- Hugging Face for transformer architectures
- The broader deep learning community for methodological insights 