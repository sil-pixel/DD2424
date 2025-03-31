import pickle
import numpy as np

import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Load a batch of training data
cifar_dir = 'Datasets/cifar-10-batches-py/'
with open(cifar_dir + 'data_batch_1', 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
# Extract the image data and cast to float from the dict dictionary
X = dict[b'data'].astype(np.float64) / 255.0
X = X.transpose()
nn = X.shape[1]

# Reshape each image from a column vector to a 3d array
X_im = X.reshape((32, 32, 3, nn), order='F')
X_im = np.transpose(X_im, (1, 0, 2, 3))  # Fix transpose operation
# Display the first 5 images
ni = 5
fig, axs = plt.subplots(1, 5, figsize=(10, 5))
for i in range(ni):
    axs[i].imshow(X_im[:, :, :, i])  # Fix incorrect name 'X im'
    axs[i].axis('off')  # Fix improper quotation marks

plt.pause(0.1)
