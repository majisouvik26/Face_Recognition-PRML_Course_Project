# Face Recognition using Combined Features

This code demonstrates a face recognition system that uses a combination of Histogram of Oriented Gradients (HOG), Local Binary Patterns (LBP), and pre-trained Convolutional Neural Network (CNN) features. The features are extracted from images and used to train a Logistic Regression classifier for recognizing faces.

## Prerequisites

- Python 3.x
- OpenCV
- Scikit-image
- Scikit-learn
- PyTorch
- Torchvision

## Dataset

The code uses the [Labeled Faces in the Wild (LFW) dataset]([http://vis-www.cs.umass.edu/lfw/](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)). You can download the dataset from the provided Kaggle link or use your own dataset.

## Feature Extraction

The code defines the following functions for feature extraction:

1. `extract_features(image_path, model)`: Extracts features from the given image using a pre-trained ResNet-50 model.
2. `compute_hog(img)`: Computes the HOG features from the given image.
3. `calcLBP(img)`: Computes the LBP features from the given image.

## Training

The code trains a Logistic Regression classifier using a combination of HOG, LBP, and CNN features extracted from the LFW dataset. The trained model is saved as a pickle file (`demo1.pkl`).

## Inference

The `infer(image_path)` function in the DEMO_CODE.ipynb takes an image path as input and performs the following steps:

1. Reads the image.
2. Extracts HOG, LBP, and CNN features from the image.
3. Combines the features into a single feature vector.
4. Loads the trained model from the pickle file named- demo.pkl.
5. Uses the trained model to predict the person's identity in the given image.

## Usage

For pkl file(model) generation, you can use pkl_generation_code.ipynb and make slight changes as per your convenience and usage.

1. Make sure you have the required dependencies installed.
2. Download the LFW dataset.
3. Run the code to train the model and generate the pickle file.

Now, for testing the model on a new image you can use DEMO_CODE.ipynb.

1. Run the `infer(image_path)` function and provide the path to the image file.
2. I have given some exaple images in the folders named `Ariel Sharon Dataset` & `George W Bush Dataset`
3. This code works with almost 96% accuracy

Example:

```python
prediction = infer('path/to/image.jpg')
print("Prediction:", prediction)
```

## Limitations

- The code is designed to work with the LFW dataset.
- `demo.pkl` is trained on LFW dataset (only persons with greater than 70 images in the dataset). 
- It can predict only those persons on which it is trained. For training it with more persons, you can make chnages in `pkl_generation_code.ipynb` file.
- Only images with two spatial dimensions are supported. If using with color/multichannel images, specify `channel_axis` in `pkl_generation_code.ipynb` file.
- You must make sure that your image provided as input must be of these persons- Ariel Sharon, Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Hugo Chavez, Tony Blair.
- The performance of the model may vary depending on the quality and diversity of the dataset used for training.
