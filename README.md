# Rainbow-Image-Clustering

This project explores image analysis and clustering techniques using a vibrant rainbow image. It transforms pixel data (coordinates and RGB values) into a dataset and applies clustering algorithms like scikit-learn's KMeans and a custom PyTorch implementation. The project demonstrates practical image processing, feature normalization, and the visual reconstruction of clustered data to reveal underlying patterns.

## Project Overview

The objective of this project is to bridge image processing and data analysis by transforming a visual image into a structured dataset. By treating each pixel as a distinct data point with spatial (x, y) and color (Red, Green, Blue) features, we apply unsupervised machine learning algorithms to group similar pixels and reconstruct a segmented version of the original image.

## Key Tasks and Methodology

### 1. Dataset Creation and Preprocessing
* **Image Loading:** Utilizes the PIL (Pillow) library to read the target image, ensuring any alpha channels (RGBA) are converted to standard RGB formats.
* **Data Extraction:** Converts the image into a NumPy array to systematically extract the x-coordinate, y-coordinate, and R, G, B values for every individual pixel.
* **Normalization:** Scales the extracted features to ensure balanced weighting during the distance calculations in the clustering phase.
* **Data Structuring:** Compiles the processed features into a comprehensive Pandas DataFrame for streamlined analysis.

### 2. Custom PyTorch Clustering Implementation
Beyond standard libraries, this project features a custom K-Means clustering algorithm built from scratch using PyTorch tensors to process the image features (x, y, R, G, B).
* **Initialization:** Selects 8 random data points from the dataset to serve as initial cluster centroids.
* **Distance Calculation:** Computes the pairwise Euclidean distance between all pixels and the current centroids using `torch.cdist`.
* **Iterative Updates:** Assigns each pixel to the nearest centroid and updates the centroid positions by calculating the mean of all assigned pixels.
* **Convergence:** The loop halts early if the shift in centroid positions falls below a defined threshold (1e-4).

### 3. Image Reconstruction
* Appends the final cluster labels back to the original DataFrame.
* Rescales the assigned cluster RGB values back to the standard [0, 255] color range.
* Maps each pixel to its newly assigned cluster color and reshapes the array back into the original image dimensions to visualize the final segmented output using Matplotlib.

## Technical Challenges & Solutions

* **Empty Clusters:** During the iterative centroid update phase, certain clusters could theoretically lose all assigned pixels. This was mitigated by retaining the previous iteration's centroid coordinates for any empty clusters, preventing computation errors.
* **Training Efficiency:** Implementing a strict convergence threshold (difference < 1e-4) prevented unnecessary compute cycles once the cluster boundaries stabilized.

## Technologies Used
* Python (Jupyter Notebook)
* PyTorch (Custom algorithm implementation and tensor operations)
* scikit-learn (Baseline KMeans implementation)
* NumPy & Pandas (Data manipulation and structuring)
* PIL / Pillow (Image loading and processing)
* Matplotlib (Data and image visualization)
