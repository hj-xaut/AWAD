import hdf5storage
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib
from eval import *
import scipy.io as scio

matplotlib.use('TkAgg')

path = r'F:\数据集\WDC\wdc4x_input32\test\1.mat'
mat = scio.loadmat(path, verify_compressed_data_integrity=False)
label = torch.from_numpy(mat['label'].astype(np.float32))
lr = torch.from_numpy(mat['lr'].astype(np.float32)).unsqueeze(0)

lrsr = F.interpolate(lr, scale_factor=4, mode='bicubic').squeeze(0)
scio.savemat(r'.\2.mat', {'SR': lrsr, 'label': label})

# plt.imshow(label[0,:,:],cmap='gray')
# plt.show()
# plt.imshow(lrsr[0,:,:],cmap='gray')
# plt.show()

# import scipy.io as scio
# import cv2
# import os
# import numpy as np
#
# # Path to the .mat file
# path = r'E:\dataset\pavia4x_input32\test\00001.mat'
# # Load the .mat file
# mat = scio.loadmat(path)
#
# # Extract the label data
# label = mat['label']
#
# # Get the shape of the 3D array
# c, h, w = label.shape
#
# # Create a directory to save the images if it doesn't exist
# output_dir = 'output_images'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# # Iterate through each slice of the 3D array and save as grayscale images
# for i in range(c):
#     # Normalize the data to range 0-255
#     img = label[i, :, :].astype(np.float64)  # Convert to float64 for precision
#     img = 255 * (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize to 0-255
#     img = img.astype(np.uint8)  # Convert to uint8 for saving as image
#
#     # Construct the output file path
#     output_file = os.path.join(output_dir, f'{i}.png')
#     # Save the image
#     cv2.imwrite(output_file, img)
#
# print(f"Saved {c} images in '{output_dir}' directory.")

