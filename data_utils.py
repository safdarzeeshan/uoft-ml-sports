from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


class ImagePreprocessor:
    """
    A helper class for preprocessing image files into resized grayscale images.

    This class handles loading, resizing, RGB conversion, and grayscale transformation.

    Parameters:
    - target_shape: Desired output image size (height, width)
    - base_path: Base directory for image paths in the DataFrame
    """
    def __init__(self, target_shape=(224, 224), base_path="data"):
        self.target_shape = target_shape
        self.base_path = base_path

    def load_and_preprocess(self, df_subset):
        """
        Loads and preprocesses images from a DataFrame with relative file paths.

        Parameters:
        - df_subset: DataFrame with a 'filepaths' column containing image paths

        Returns:
        - images: List of resized grayscale images
        - valid_labels: List of labels corresponding only to successfully loaded images
        """
        images = []
        valid_labels = []

        for _, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
            img_path = os.path.join(self.base_path, row['filepaths'])

            if not img_path.lower().endswith(".jpg"):
                continue

            try:
                image = imread(img_path)

                if image.ndim == 2:
                    image = np.stack((image,) * 3, axis=-1)
                elif image.shape[2] == 4:
                    image = image[:, :, :3]
                elif image.shape[2] != 3:
                    continue

                image = resize(image, (*self.target_shape, 3), anti_aliasing=True)
                image_gray = rgb2gray(image)

                images.append(image_gray)
                valid_labels.append(row['labels'])

            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

        return images, valid_labels


def plot_confusion_matrix(y_true, y_pred, class_labels, figsize=(16, 14), title="Confusion Matrix", normalize=False):
    """
    Plots a labeled confusion matrix using seaborn's heatmap.

    Parameters:
    - y_true: Ground truth labels
    - y_pred: Predicted labels
    - class_labels: List of class label names
    - figsize: Size of the plot
    - title: Plot title
    - normalize: Whether to normalize the matrix values
    """
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
