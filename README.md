# Plant Disease Classification with Deep Learning 🌱🤖

Welcome to the **Plant Disease Classification** project! This repository contains a Jupyter Notebook that leverages **Convolutional Neural Networks (CNNs)** to classify plant diseases using image datasets.

---

## Table of Contents 📖

1. [Introduction](#introduction)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Model Details](#model-details)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Future Work](#future-work)
9. [Acknowledgments](#acknowledgments)

---

## Introduction 🌾

This project focuses on identifying various plant diseases using machine learning techniques. By training a CNN model, we aim to accurately classify images of plants into their respective disease categories.

---

## Features ✨

- **Deep Learning with TensorFlow/Keras**: Utilizes state-of-the-art libraries for efficient model training.
- **Data Augmentation**: Enhances the dataset with transformations to improve model generalization.
- **GPU Acceleration**: Leverages available GPUs for faster computations.
- **Multi-class Classification**: Handles 38 distinct plant classes, including healthy plants.

---

## Dataset 📊

The dataset contains images of plants categorized into 38 classes, including healthy and diseased plants.

- **Train Set**: 70,295 images
- **Validation Set**: 17,572 images

The images are sourced from the "New Plant Diseases Dataset (Augmented)" and organized into training and testing directories.

---

## Model Details 🧠

The notebook implements a **Convolutional Neural Network (CNN)** using TensorFlow and Keras. Key features include:

- **Input Size**: 224x224 pixels
- **Data Generators**: ImageDataGenerator for real-time data augmentation
- **Layers**: Multiple convolutional layers followed by fully connected layers

---

## Installation ⚙️

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Abdelrhman-Ellithy/Plant-Disease-CNN-Deep-Learning.git
   ```
3. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

---

## Usage 🚀

1. **Dataset Preparation**: Ensure the dataset is structured into `train` and `valid` directories.
2. **Run the Notebook**: Execute the cells step by step to train the model.
3. **Results**: View accuracy and loss plots, and evaluate the model on test images.

---

## Results 📈

- **Accuracy**: Achieves high accuracy on the validation set (details in notebook).
- **Confusion Matrix**: Visualizes model performance across classes.
- **Sample Predictions**: Displays model predictions on sample images.

---

## Future Work 🔮

- **Model Optimization**: Experiment with architectures like ResNet or EfficientNet.
- **Hyperparameter Tuning**: Adjust learning rates, batch sizes, and optimizers.
- **Deployment**: Create a web app or mobile app for real-time disease detection.

---

## Acknowledgments 🙏

Special thanks to:

- [Dataset Providers](https://www.kaggle.com/datasets): For making the plant dataset available.
- TensorFlow/Keras Community: For robust tools and support.

---

## License 📜

This project is licensed under the MIT License. Feel free to use and modify the code as per your needs.

---

Happy Coding! 🌟