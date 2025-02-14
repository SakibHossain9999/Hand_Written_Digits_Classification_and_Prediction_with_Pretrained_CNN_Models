# Hand-Written-Digits-Classification-and-Prediction-with-Pretrained-CNN-Models

This repository contains the implementation of five deep learning models trained for Bengali digits classification. The models utilize transfer learning with well-known convolutional neural networks (CNNs) pretrained on ImageNet.

## Models Included

* VGG19

* ResNet50

* InceptionV3

* EfficientNetB7

* DenseNet201

## Dataset

[Bengali Handwritten Digit Dataset](https://www.kaggle.com/datasets/wchowdhu/bengali-digits)

The dataset is structured into 10 folders, each folder contains images of Bengali digits for the corresponding number,
0 --> ০

1 --> ১

2 --> ২

3 --> ৩

4 --> ৪

5 --> ৫

6 --> ৬

7 --> ৭

8 --> ৮

9 --> ৯

## Preprocessing

Preprocessing steps such as resizing, normalization, and augmentation have been applied to enhance model performance.

## Repository Structure

* models/ - Contains trained models in .h5 format.

* notebooks/ - Jupyter notebooks for training and evaluation.

* requirements.txt - Dependencies for running the code.

## How to Use

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/SakibHossain9999/Hand-Written-Digits-Classification-and-Prediction-with-Pretrained-CNN-Models.git
    cd Hand-Written-Digits-Classification-and-Prediction-with-Pretrained-CNN-Models
    ```

2. **Install Dependencies**:
    Install the required Python packages by running:
    ```bash
    pip install -r requirements.txt
    ```
    
3. **Run the .ipynb of any deep learning model(s), but make sure you have dataset in the same directory of your model.ipynb file(s)**

## Results

Evaluation metrics such as accuracy, precision, recall, and F1-score are used to compare model performance. Some visualization of the results can be found in the results/ directory.

## Acknowledgments

This project utilizes TensorFlow/Keras for deep learning model training and inference.

## Licensing

This project is licensed under the MIT License, a permissive open-source license that allows others to use, modify, and distribute the project's code with very few restrictions. This license can benefit research by promoting collaboration and encouraging the sharing of ideas and knowledge. With this license, researchers can build on existing code to create new tools, experiments, or projects, and easily adapt and customize the code to suit their specific research needs without worrying about legal implications. The open-source nature of the MIT License can help foster a collaborative research community, leading to faster innovation and progress in their respective fields. Additionally, the license can help increase the visibility and adoption of the project, attracting more researchers to use and contribute to it.



