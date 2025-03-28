from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

class HOGTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that extracts HOG (Histogram of Oriented Gradients) features
    from preprocessed and resized grayscale images.

    Parameters:
    - orientations: Number of orientation bins for HOG
    - pixels_per_cell: Size (in pixels) of a cell
    - cells_per_block: Number of cells per block
    - block_norm: Normalization method for blocks
    """

    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys'):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X=None, y=None):
        """
        No fitting necessary for HOG extraction.
        Returns self.
        """
        return self

    def transform(self, X, y=None):
        """
        Applies HOG transformation to a list or array of grayscale images.

        Parameters:
        - X: Array-like of preprocessed grayscale images

        Returns:
        - X_transformed: Numpy array of HOG features
        """
        X_transformed = []
        for image in tqdm(X):
            features = hog(
                image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm
            )
            X_transformed.append(features)

        return np.array(X_transformed)
