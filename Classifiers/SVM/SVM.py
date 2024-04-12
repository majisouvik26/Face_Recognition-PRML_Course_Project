import os
from zipfile import ZipFile
import os
import cv2
import kaggle
import numpy as np
from skimage.io import imread
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

os.environ['KAGGLE_USERNAME'] = "veerarajuelluru"
os.environ['KAGGLE_KEY'] = "30fa741bb75b421e48f7c536441f6a85"

# pip install kaggle
# kaggle datasets download -d jessicali9530/lfw-dataset


# with ZipFile('lfw-dataset.zip', 'r') as zip_ref:
#     zip_ref.extractall('lfw-dataset')
#
#
# os.remove('lfw-dataset.zip')


def extract_features(image_path, model):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    features = features.squeeze(0)
    return features


# def compute_hog(img):
#     resized_img = resize(img, (128*4, 64*4))
#     fd, hog_image = hog(resized_img,
#                         orientations=9,
#                         pixels_per_cell=(8, 8),
#                         cells_per_block=(2, 2),
#                         visualize=True
#                         )
#     return fd


def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value


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


resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

lfw_folder = r'C:\Users\Veeraraju_elluru\Desktop\Veeraraju\IITJ\Sophomore Year\Sem_2\PRML\Project\content\lfw-dataset\lfw-funneled\lfw_funneled'
X, y = [], []
for folder_name in os.listdir(lfw_folder):
    folder_path = os.path.join(lfw_folder, folder_name)
    if os.path.isdir(folder_path):
        num_images = len(os.listdir(folder_path))
        if num_images > 70:
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                image = imread(image_path)
                lbp_feature = calcLBP(image)
                cnn_feature = extract_features(image_path, resnet).numpy()
                lbp_feature = lbp_feature.reshape(-1)
                cnn_feature = cnn_feature.flatten()
                combined_feature = np.concatenate((lbp_feature, cnn_feature))

                X.append(combined_feature)
                y.append(folder_name)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

parameters = [
        {"kernel":
            ["linear"],
            "C": [0.0001, 0.001, 0.1, 1, 10, 100, 1000]},
        {"kernel":
            ["poly"],
            "degree": [2, 3, 4],
            "C": [0.0001, 0.001, 0.1, 1, 10, 100, 1000]},
        {"kernel":
            ["rbf"],
            "gamma": ["auto", "scale"],
            "C": [0.0001, 0.001, 0.1, 1, 10, 100, 1000]}
]

svm_clf = SVC()
print("tuning hyperparameters via grid search")
grid = GridSearchCV(estimator=SVC(), param_grid=parameters, n_jobs=-1)
grid.fit(X_train,y_train)
print(f"grid search best score: {grid.best_score_ * 100:.2f}%")
print(f"grid search best parameters: {grid.best_params_}")

model = grid.best_estimator_
y_pred_svm = model.predict(X_test)
print(classification_report(y_test, y_pred_svm))
print("SVM Classifier")
print("Classification Report:")
print(classification_report(y_test_encoded, y_pred_svm, target_names=label_encoder.classes_))

