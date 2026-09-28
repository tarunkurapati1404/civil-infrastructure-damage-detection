# Model Directory

This directory contains the YOLOv4 model configuration and model-related files used for civil infrastructure damage (pothole) detection.

## Model Specifications

* **Architecture:** YOLOv4
* **Framework:** Darknet
* **Detection Class:** Pothole
* **Input Resolution:** $416 \times 416$
* **Training Batches / Iterations:** $2000$

## Trained Weights

> **Note:** The final trained YOLOv4 weights file is not included directly in this repository due to its large file size.

The original trained model weights were generated and saved as:

`yolov4_training_final.weights`

For instructions on how to set up the configuration, download pre-trained weights, and train the model from scratch, please refer to the project Jupyter notebook located in the `notebooks/` directory.