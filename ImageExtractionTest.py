import numpy as np
from PIL import Image


img1 = np.asarray(Image.open("14_59_51_frame_0.tif"))
img2 = np.asarray(Image.open("14_59_51_frame_1.tif"))

# Create a boolean mask for different pixels
# This assumes both images have the same shape
mask = img1 != img2

# Find the coordinates of the different pixels (returns a list of [y, x] coordinates)
# If you want to use a threshold for minor differences, you can sum the absolute differences and compare to a threshold
# Example for threshold: mask = np.sum(np.abs(img1 - img2), axis=-1) > threshold
coords = np.argwhere(mask)

# Print or process the coordinates
print(f"Found {len(coords)} different pixels.")
# print(coords) 

print(coords)

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import skimage as ski

matplotlib.rcParams['font.size'] = 18

moon = ski.io.imread('14_59_51_frame_0.tif')
moon2 = ski.io.imread('14_59_51_frame_1.tif')
edges = ski.filters.sobel(moon)
edges2 = ski.filters.sobel(moon2)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# Display first image
axes[0].imshow(edges, cmap='gray')
axes[0].set_title("Image 1")
axes[0].axis('off')  # Hide axis labels

# Display second image
axes[1].imshow(edges2, cmap='gray')
axes[1].set_title("Image 2")
axes[1].axis('off')

plt.tight_layout()
plt.show()
