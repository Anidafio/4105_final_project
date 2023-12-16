import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import os

# Function to extract HOG features from an image
def extract_features(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate HOG features
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
    return hog_features

# Load images from directories (adjust paths as per your dataset structure)
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Path to folders containing images of power poles and trees
power_pole_train_folder = 'C:\\Users\\Nurlan\Documents\\vs code\\ml_project\\train\\poles'
streetlight_train_folder = 'C:\\Users\\Nurlan\Documents\\vs code\\ml_project\\train\\streetlight'
tree_train_folder = 'C:\\Users\\Nurlan\Documents\\vs code\\ml_project\\train\\trees'
power_pole_val_folder = 'C:\\Users\\Nurlan\Documents\\vs code\\ml_project\\val\\poles'
streetlight_val_folder = 'C:\\Users\\Nurlan\Documents\\vs code\\ml_project\\val\\streetlight'
tree_val_folder = 'C:\\Users\\Nurlan\Documents\\vs code\\ml_project\\val\\trees'

# Load images and extract features
power_pole_train_images = load_images_from_folder(power_pole_train_folder)
streetlight_train_images = load_images_from_folder(streetlight_train_folder)
tree_train_images = load_images_from_folder(tree_train_folder)
power_pole_val_images = load_images_from_folder(power_pole_val_folder)
streetlight_val_images = load_images_from_folder(streetlight_val_folder)
tree_val_images = load_images_from_folder(tree_val_folder)

# Label images
power_pole_train_labels = np.zeros(len(power_pole_train_images))  # 0 for power poles
tree_train_labels = np.ones(len(tree_train_images))  # 1 for trees
streetlight_train_labels = np.full(len(streetlight_train_images), 2)
power_pole_val_labels = np.zeros(len(power_pole_val_images))  # 0 for power poles
tree_val_labels = np.ones(len(tree_val_images))  # 1 for trees
streetlight_val_labels = np.full(len(streetlight_val_images), 2)

# Concatenate images and labels
X_train = np.array(power_pole_train_images + tree_train_images + streetlight_train_images)
y_train = np.concatenate((power_pole_train_labels, tree_train_labels, streetlight_train_labels))
X_test = np.array(power_pole_val_images + tree_val_images + streetlight_val_images)
y_test = np.concatenate((power_pole_val_labels, tree_val_labels, streetlight_val_labels))

# Extract HOG features for all images
features = []
for image in X_train:
    feature = extract_features(image)
    features.append(feature)
X_train_features = np.array(features)

features = []
for image in X_test:
    feature = extract_features(image)
    features.append(feature)
X_val_features = np.array(features)

# Create SVM classifier
clf = svm.SVC(kernel='linear')

# Train the classifier
clf.fit(X_train_features, y_train)

# Predict on the test set
y_pred = clf.predict(X_val_features)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix, classification_report

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Classification report
class_report = classification_report(y_test, y_pred)

# Print additional metrics and visualizations
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)