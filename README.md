# Handwritten Bengali Digit Classification with Pretrained CNN Models

Transfer learning for recognizing handwritten Bengali digits (০–৯). This repository trains and compares five ImageNet-pretrained convolutional neural networks and includes a small Flask web app for predicting the digit in an uploaded image.

## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Training and Evaluation](#training-and-evaluation)
- [Web App](#web-app)
- [Results](#results)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Overview

Bengali is written in its own script, and handwritten digits vary a lot from person to person. Instead of training a network from scratch, this project fine-tunes well-known CNN architectures that were pretrained on ImageNet, then evaluates how each one performs on Bengali digit images. The best-performing trained model is served through a Flask app so you can upload an image and get a predicted digit back.

## Models

| Model          | Pretrained on |
| -------------- | ------------- |
| VGG19          | ImageNet      |
| ResNet50       | ImageNet      |
| InceptionV3    | ImageNet      |
| EfficientNetB7 | ImageNet      |
| DenseNet201    | ImageNet      |

All models are built with TensorFlow/Keras.

## Dataset

The project uses the [Bengali Handwritten Digit Dataset](https://www.kaggle.com/datasets/wchowdhu/bengali-digits) from Kaggle.

The dataset is organized into 10 folders, one per digit class:

| Folder | Digit |
| :----: | :---: |
| 0      | ০     |
| 1      | ১     |
| 2      | ২     |
| 3      | ৩     |
| 4      | ৪     |
| 5      | ৫     |
| 6      | ৬     |
| 7      | ৭     |
| 8      | ৮     |
| 9      | ৯     |

The dataset is not included in this repository. Download it from Kaggle and place it next to the notebook you plan to run (see [Training and Evaluation](#training-and-evaluation)).

## Preprocessing

Images are resized, normalized, and augmented before training to improve generalization. The web app applies the matching resize and normalization at prediction time (224 × 224 pixels, pixel values scaled to the 0–1 range).

## Repository Structure

```
.
├── Evaluation Graphs/    # Evaluation plots for the trained models
├── Jupyter Notebooks/    # Training and evaluation notebooks
├── templates/            # HTML templates for the Flask app
├── app.py                # Flask app for digit prediction
├── requirements.txt      # Python dependencies
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/SakibHossain9999/Hand_Written_Digits_Classification_and_Prediction_with_Pretrained_CNN_Models.git
cd Hand_Written_Digits_Classification_and_Prediction_with_Pretrained_CNN_Models
```

### 2. Install dependencies

Using a virtual environment is recommended.

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Training and Evaluation

1. Download the [dataset](https://www.kaggle.com/datasets/wchowdhu/bengali-digits) from Kaggle and extract it.
2. Open a notebook from the `Jupyter Notebooks/` folder.
3. Make sure the dataset folder sits in the same directory as the notebook you are running.
4. Run the notebook to train and evaluate the model.

Each model can be run independently, so you only need to run the ones you are interested in.

## Web App

The Flask app (`app.py`) loads a trained InceptionV3 model and predicts the digit in an uploaded image.

The app expects the trained model file at:

```
Trained Models/ImageNet_InceptionV3_bengali_digits.h5
```

Train the model with the corresponding notebook and save it to that path, or update the path in `app.py` to point to your own model file.

### Run the app

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser and upload an image of a handwritten Bengali digit.

### API

The app also exposes a prediction endpoint.

**`POST /predict`**

| Field | Type | Description                  |
| ----- | ---- | ----------------------------- |
| file  | file | Image of a handwritten digit |

Example:

```bash
curl -X POST -F "file=@digit.png" http://127.0.0.1:5000/predict
```

Response:

```json
{ "prediction": "5" }
```

The returned value is the predicted class, which matches the folder number in the dataset (for example, `5` corresponds to ৫). If no file is sent, the endpoint returns a `400` error with a message.

## Results

Models are compared using accuracy, precision, recall, and F1-score. Evaluation plots for the trained models are available in the [`Evaluation Graphs`](Evaluation%20Graphs) folder.

## Acknowledgments

- Built with [TensorFlow/Keras](https://www.tensorflow.org/) and [Flask](https://flask.palletsprojects.com/).
- Pretrained weights are from the ImageNet-trained models provided by Keras.
- Dataset: [Bengali Handwritten Digit Dataset](https://www.kaggle.com/datasets/wchowdhu/bengali-digits) on Kaggle.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute the code with minimal restrictions.
