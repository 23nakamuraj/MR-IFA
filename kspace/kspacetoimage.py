#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:44:46 2024

@author: koeinakamura
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Load the .mat file
mat = scipy.io.loadmat('/Users/koeinakamura/Documents/MATLAB/MR-IFA_MATLAB/phantom_data.mat')

# Extract k-space data
kData = mat['kData']
karray = np.array(kData)

# Number of slices
num_slices = karray.shape[2]

# Function to process a single slice
def process_slice(karray, slice_index):
    # Extract the slice
    slice_data = karray[:, :, slice_index]

    # Perform IFFT on the slice
    image_data = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(slice_data)))
    
    # Get the magnitude image
    image_data = np.abs(image_data)
    
    # Normalize the image to the range [0, 1]
    image_data = image_data - np.min(image_data)
    image_data = image_data / np.max(image_data)
    
    return image_data

# Process and display each slice
for slice_index in range(num_slices):
    image_data = process_slice(karray, slice_index)
    
    # Display the image
    plt.figure()
    plt.imshow(image_data, cmap='gray')
    plt.title(f'Slice {slice_index + 1}')
    plt.axis('off')
    plt.show()
