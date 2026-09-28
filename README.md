# Civil Infrastructure Damage Detection Using YOLOv4

## MSc Final Project

A deep learning and computer vision project focused on detecting potholes in road images using the YOLOv4 object detection framework.

## Project Overview

The project uses the Darknet implementation of YOLOv4 to configure and train a custom object detection model for pothole detection.

The core workflow includes:

* Preparing road-image data for object detection

* Configuring YOLOv4 for a single **Pothole** class

* Preparing YOLO training configuration and data files

* Using pre-trained YOLOv4 convolutional weights

* Training the custom YOLOv4 model

* Saving the trained model weights for further use

## Technology Stack

* **Languages & Frameworks:** Python, YOLOv4, Darknet, OpenCV

* **Hardware & Acceleration:** CUDA, cuDNN, Google Colab

* **Environment:** Jupyter Notebook

## Project Workflow

```
Road Images
     ↓
Dataset Preparation
     ↓
YOLOv4 Configuration
     ↓
Training Data Preparation
     ↓
Pre-trained Weights
     ↓
YOLOv4 Model Training
     ↓
Trained Model

```

## YOLOv4 Configuration

The model was configured with the following parameters:

* **Detection class:** Pothole

* **Input size:** $416 \times 416$

* **Maximum training batches:** $2000$

* **GPU acceleration:** Enabled

* **OpenCV:** Enabled

* **cuDNN:** Enabled

## Repository Structure

```
civil-infrastructure-damage-detection/
│
├── notebooks/
│   └── MSc_Final_Project.ipynb
│
├── dataset/
├── model/
├── results/
├── src/
├── docs/
├── .gitignore
└── README.md

```

## Notebook & Google Colab

The complete project workflow, implementation details, and training steps are documented in `notebooks/MSc_Final_Project.ipynb`.

The original project was developed and trained using Google Colab with GPU acceleration. You can open and run the notebook directly in Google Colab using your workspace setup.

## Dataset

The project uses road-surface imagery containing pothole examples and corresponding object-detection annotations.

> **Note:** The full dataset is not included directly in this repository due to its large size. 

## References

The original notebook contains the complete project references, academic sources, and supporting resources used during development.

\[Open the project in Google Colab\](https://colab.research.google.com/drive/1H-uyLs7VigS6RlSiiOwVA2UFBfcVDY8d)