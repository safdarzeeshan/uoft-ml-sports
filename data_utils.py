from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
from tqdm import tqdm 
import numpy as np
from skimage.io import imread
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(df_subset, base_path="data", target_shape=(224, 224, 3)):
    X, y = [], []
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
        img_path = os.path.join(base_path, row['filepaths'])
        
        if not img_path.lower().endswith(".jpg"):
            continue
        
        try:
            image = imread(img_path)
            
            # Force RGB
            if image.ndim == 2:
                # Grayscale -> RGB
                image = np.stack((image,) * 3, axis=-1)
            elif image.shape[2] == 4:
                # RGBA -> RGB
                image = image[:, :, :3]
            elif image.shape[2] != 3:
                print(f"Skipping image with unexpected shape: {img_path}")
                continue
            
            # Resize to target shape
            image = resize(image, target_shape, anti_aliasing=True)
            
            X.append(image.flatten())  # flatten to 1D
            y.append(row['labels'])
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
            
    return np.array(X), np.array(y)


def load_data_hog(df_subset, base_path="data", target_shape=(224, 224)):
    X, y = [], []
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
        img_path = os.path.join(base_path, row['filepaths'])

        if not img_path.lower().endswith(".jpg"):
            continue

        try:
            image = imread(img_path)

            # Handle grayscale and RGBA
            if image.ndim == 2:
                image = np.stack((image,) * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
            elif image.shape[2] != 3:
                continue

            image = resize(image, (*target_shape, 3), anti_aliasing=True)
            image_gray = rgb2gray(image)

            features = hog(
                image_gray,
                orientations=8,
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
                block_norm='L2-Hys'
            )

            X.append(features)
            y.append(row['labels'])

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

    return np.array(X), np.array(y)

def plot_confusion_matrix(y_true, y_pred, class_labels, figsize=(16, 14), title="Confusion Matrix", normalize=False):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # replace NaNs from divide-by-zero

    plt.figure(figsize=figsize)
    sns.heatmap(cm, xticklabels=class_labels, yticklabels=class_labels, cmap="Blues", 
                square=True, cbar=True, linewidths=0.5)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
