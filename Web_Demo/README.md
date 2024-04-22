# Face Identification App

This is a Streamlit app that uses a pre-trained model to identify faces in uploaded images. The app combines features from Histogram of Oriented Gradients (HOG), Local Binary Patterns (LBP), and a pre-trained Convolutional Neural Network (CNN) to perform face recognition.

## Prerequisites

Before running the app, you need to set up a virtual environment and install the required dependencies. For example, you can follow these steps:

1. Install Anaconda or Miniconda if you haven't already.

2. Create a new virtual environment with Python 3.9:

```
conda create -p venv python==3.9 -y
```

3. Activate the virtual environment:

```
conda activate venv
```

4. Install the required dependencies from the `requirements.txt` file:

```
pip install -r requirements.txt
```

## Running the App

1. Make sure your virtual environment is activated.

2. Run the Streamlit app using the following command:

```
streamlit run app.py
```

3. The app will open in your default web browser. If it doesn't open automatically, you can access it by clicking the URL provided in the terminal.

## Using the App

1. Upload an image file (PNG format).
2. The app will display the uploaded image.
3. After a few seconds, the app will classify the face in the image and display the prediction.

## Files

- `app.py`: The main Streamlit application code.
- `requirements.txt`: A list of required Python packages and their versions.
- `demo.pkl`: A pre-trained model used for face recognition.

## Notes

- The app assumes that you have a pre-trained model (`demo.pkl`) available in the same directory as `app.py`. If you don't have this file, the app will not work.
- The app is designed to work with PNG image files. Other image formats are not be supported.
- You must make sure that your image provided as input must be of these persons- Ariel Sharon, Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Hugo Chavez, Tony Blair.
- Only images with two spatial dimensions are supported.
- The performance of the app depends on the quality and diversity of the dataset used to train the pre-trained model.

## Acknowledgments

This app is built using various open-source libraries and resources, including Streamlit, OpenCV, Scikit-image, PyTorch, and Torchvision.
