import os
import streamlit as st
import cv2
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle

# Load the pretrained resnet-50 model
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# Load the demo model
with open('demo.pkl', 'rb') as f:
    demo = pickle.load(f)

# defining some of the important functions that are needed 

# defining feature extraction for resnet-50
def extract_features(image, model):
    image_np = np.array(image)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image_np)
    image = image.unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    features = features.squeeze(0)
    return features


# hog calculation
def compute_hog(img):
    resized_img = resize(img, (128*4, 64*4))
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True,channel_axis=-1)
    return fd

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

# calculate lbp

def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))
    val_ar.append(get_pixel(img, center, x, y+1))
    val_ar.append(get_pixel(img, center, x+1, y+1))
    val_ar.append(get_pixel(img, center, x+1, y))
    val_ar.append(get_pixel(img, center, x+1, y-1))
    val_ar.append(get_pixel(img, center, x, y-1))
    val_ar.append(get_pixel(img, center, x-1, y-1))
    val_ar.append(get_pixel(img, center, x-1, y))

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val

def calcLBP(img):
    height, width, channel = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_lbp = np.zeros((height, width,3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
             img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
    return hist_lbp.flatten()

# Function to infer the class of an image

def infer(image_path):
    image = imread(image_path) # reading the image
    hog_feature = compute_hog(image) # hog features
    lbp_feature = calcLBP(image)     # lbp features
    cnn_feature = extract_features(image, resnet).numpy() #cnn features
    hog_feature = hog_feature.reshape(-1) # reshaping
    lbp_feature = lbp_feature.reshape(-1) #reshaping 
    cnn_feature = cnn_feature.flatten() # cnn features
    combined_feature = np.concatenate((hog_feature, lbp_feature, cnn_feature)) # combining all the features 
    prediction = demo.predict([combined_feature])
    return prediction[0]
# Streamlit code

st.title("Face Identification")
uploaded_file = st.file_uploader("Choose an image...", type=["png"])

if uploaded_file is not None:
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getvalue())
    image = Image.open("temp_image.png")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    prediction = infer("temp_image.png")
    st.write("Prediction:", prediction)
