# Face Recognition: PRML Course Project

This repository contains the code, reports, and related resources for a face recognition project undertaken as part of the Pattern Recognition and Machine Learning (PRML) course. The project aims to explore different feature extraction techniques and machine learning models for the task of face recognition.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Repository Structure](#repository-structure)
- [Reports](#reports)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The primary goal of this project is to investigate various feature extraction methods, such as Histogram of Oriented Gradients (HOG), Local Binary Patterns (LBP), and Convolutional Neural Networks (CNNs), and their combinations for face recognition. Additionally, the project explores the performance of different machine learning classifiers, including K-Nearest Neighbors (KNN), Logistic Regression, Multi-Layer Perceptron (MLP), Naive Bayes, Support Vector Machines (SVM), and tree-based algorithms like AdaBoost, XGBoost, Decision Trees, and Random Forests.

## Dataset

The project uses the [Labeled Faces in the Wild (LFW) dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) for training and evaluation. The dataset contains over 13,000 images of faces collected from the web, with a focus on capturing variations in pose, illumination, and background.

## Prerequisites

Before running the code, ensure that you have the following dependencies installed:

- Python 3.x
- NumPy
- OpenCV
- Scikit-image
- Scikit-learn
- PyTorch
- Torchvision

## Repository Structure

The repository is organized as follows:

```
Face_Recognition-PRML_Course_Project/
│
├── classifiers/
│   ├── KNN/
│   ├── Logistic_Regression/
│   ├── MLP/
│   ├── Naive_Bayes/
│   ├── SVM/
│   └── Tree_based_algo/
│       ├── AdaBoost/
│       ├── Decision_Tree/
│       ├── Random_Forest/
│       └── Xgboost/
├── Demo_Code/
│   ├── Ariel Sharon Dataset/
│   ├── GEORGE W BUSH Dataset/
│   ├── DEMO_CODE.ipynb
│   ├── pkl_generation_code.ipynb
│   ├── demo.pkl
│   └── README.md
├── Web_Demo/
│   ├── app.py
│   ├── demo.pkl
│   ├── requirements.txt
│   ├── README.md
├── Mid_Report.pdf
├── PRML_Project_Report.pdf
├── Preprocessing_Part_.ipynb
├── LICENSE
└── README.md
```

- `classifiers/`: Contains subdirectories for different machine learning classifiers, each with code for various feature extraction techniques.
- `dataset/`: Empty directory for storing the LFW dataset (not included in the repository due to size constraints).
- `reports/`: Contains the mid-term and final project reports in PDF format.
- `requirements.txt`: A file listing the required Python packages and their versions.
- `Preprocessing_Part_.ipynb`: A Jupyter Notebook containing the preprocessing code.
- `README.md`: This file, providing an overview of the project and instructions for usage.

## Reports

The project includes two reports:

1. **Mid-Term Report**: Provides an overview of the project, discusses the dataset, and outlines the initial approaches and results. You can find the mid-term report [here](https://github.com/majisouvik26/Face_Recognition-PRML_Course_Project/blob/main/Mid_Report.pdf).

2. **Final Project Report**: Presents the complete details of the project, including the methodology, experimental results, analysis, and conclusions. The final report is available [here](https://github.com/majisouvik26/Face_Recognition-PRML_Course_Project/blob/main/PRML_Project_Report.pdf).

## Usage

1. Clone the repository to your local machine.
2. Navigate to the desired classifier subdirectory (e.g., `classifiers/KNN/`) and run the appropriate Python scripts or Jupyter Notebooks to test the classifiers.
3. You can find the demo code to test it on new data in the `Demo_Code` folder.
4. You can find the code of Web demo in the `Web_Demo` folder.

Note: Detailed instructions for running Demo Code & Web Demo are provided within the respective subdirectories.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License]([LICENSE](https://github.com/majisouvik26/Face_Recognition-PRML_Course_Project/blob/main/LICENSE)).
