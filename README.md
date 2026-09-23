# Forest Segmentation and Coverage Classification Using Deep Learning

This project applies deep learning and computer vision to satellite imagery for two tasks:

1. **Forest semantic segmentation** – predicting pixel-level forest masks.
2. **Forest coverage classification** – classifying images as high-forest or low/non-forest coverage.

The project uses the **Augmented Forest Segmentation Dataset** with **5,108 paired RGB satellite images and binary masks** and compares baseline CNNs with pretrained transfer-learning models.

## 1. Forest Semantic Segmentation

Compared a U-Net trained from scratch with a pretrained DeepLabV3-ResNet101 model.

The DeepLabV3 pipeline used:
- Albumentations-based data augmentation
- Weighted Dice + Focal loss
- AdamW with OneCycleLR
- Validation-based threshold tuning
- Test-time augmentation (TTA)

### Results

| Model | Test IoU | Dice | Pixel Accuracy |
|------|------:|------:|------:|
| U-Net | 0.769 | 0.869 | 0.832 |
| DeepLabV3-ResNet101 | 0.787 | 0.879 | 0.865 |
| DeepLabV3-ResNet101 + TTA | **0.793** | **0.883** | **0.867** |

The final model achieved **0.87 precision** and **0.92 recall** on the forest class.

## 2. Forest Coverage Classification

Image-level labels were derived from segmentation masks using a **50% forest-coverage threshold**.

ResNet18 and MobileNetV2 were fine-tuned using ImageNet pretrained weights, weighted cross-entropy loss, and stratified train/validation/test splits.

### Results

| Model | ROC-AUC |
|------|------:|
| ResNet18 | 0.909 |
| MobileNetV2 | **0.930** |

MobileNetV2 was selected as the preferred classifier because it achieved the highest ROC-AUC while remaining lightweight for deployment.

## Model Serving Prototype

A **FastAPI-based prototype** was developed for:
- Forest classification
- Pixel-level segmentation and coverage estimation
- Prediction logging, monitoring, and drift detection

The deployment design also includes retraining and model update strategies.

## Dataset

This repository does not include the original satellite imagery because of their size.

Dataset:  
https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation
