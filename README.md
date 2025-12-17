# PROJ_ASD_ML
**ML Project: Autism Spectrum Detection from Facial Images**

## Overview
This project develops machine learning models to detect Autism Spectrum Disorder (ASD) from facial images using transfer learning with state-of-the-art deep learning architectures.

## Problem Statement
Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition where early detection is critical for timely interventions that can significantly improve outcomes.

### Challenges with Current Diagnostic Practices
- Rely heavily on behavioral assessments
- Time-intensive evaluation processes
- Subjective interpretation
- Delayed diagnoses affecting early intervention

Recent research suggests that subtle facial morphology differences are associated with ASD in children, enabling computer vision-based screening approaches.

## Project Approach
This project implements a comprehensive transfer learning pipeline for ASD detection:

1. **Dataset Preparation**: Structured image dataset with Autistic and Non-Autistic classes
2. **Model Experimentation**: Evaluated multiple state-of-the-art architectures
3. **Two-Stage Training**: Frozen backbone training followed by fine-tuning
4. **Weighted Loss**: Penalty on false negatives to minimize missed ASD cases
5. **Threshold Optimization**: Targeting 95% recall for screening effectiveness
6. **Interpretability**: Grad-CAM visualizations for model decision explanation

## Models Evaluated
Multiple architectures were tested, with the best-performing models selected:

- Custom-built CNN models
- MobileNetV2
- VGG-19
- ResNet-50 / ResNet-50V2
- DenseNet-121
- Xception
- **EfficientNet-B0** ⭐ — Minimal false negatives, optimized for screening
- **InceptionV3** ⭐ — Strong overall performance with balanced metrics

## Repository Contents

### Jupyter Notebooks
- **[INCEPTIONV3.ipynb](INCEPTIONV3.ipynb)** — InceptionV3 implementation with 5-fold cross-validation, includes model training, evaluation, and confusion matrix analysis
- **[Transfer_Learning_Unified.ipynb](Transfer_Learning_Unified.ipynb)** — Unified pipeline comparing EfficientNet-B0, ResNet50V2, and InceptionV3 with two-stage training, weighted loss, threshold optimization, and Grad-CAM interpretability

### Key Features
- **K-Fold Cross-Validation**: Robust validation with stratified splits
- **Data Augmentation**: Horizontal flip, rotation, shifts, and zoom
- **Early Stopping & Learning Rate Reduction**: Prevent overfitting
- **Custom Metrics**: Accuracy, AUC, Precision, Recall
- **Weighted Binary Cross-Entropy**: Penalizes false negatives (crucial for medical screening)
- **Grad-CAM Visualizations**: Explainable AI showing which facial regions influence predictions

## Live Demo
Try the model inference with our Streamlit web application:
**[ASD Detection Demo App](https://asddemoapp-livelink.streamlit.app/)**

## Model Performance
The models are optimized for **high recall** (sensitivity) to minimize false negatives, which is critical for screening applications. Performance metrics include:
- Accuracy
- Recall (Sensitivity) — prioritized to catch all potential ASD cases
- Precision
- Specificity
- False Negative Rate

## Technical Stack
- **TensorFlow/Keras**: Deep learning framework
- **Transfer Learning**: ImageNet pre-trained models
- **scikit-learn**: Cross-validation and metrics
- **OpenCV & Matplotlib**: Image processing and visualization
- **Streamlit**: Web application for inference

## Usage
The notebooks are designed to run in Google Colab with Google Drive integration for dataset storage. Key configurations include:
- Image size: 224×224 (EfficientNet, ResNet) or 299×299 (InceptionV3)
- Batch sizes: 32 (stage 1), 16 (stage 2)
- Learning rates: 1e-3 (frozen), 1e-5 (fine-tuning)
- Target recall: 95% for screening effectiveness





