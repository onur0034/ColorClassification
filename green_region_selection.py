import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread(r"C:\Users\Onur\openCV_IMG_process\colorRegionApp\CableLabelRGB.bmp")


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()

# Convert to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define color ranges (HSV)
colors = {
    "Green": ([35, 50, 50], [85, 255, 255]),
    "Red": ([0, 50, 50], [10, 255, 255]),
    "Blue": ([100, 50, 50], [130, 255, 255]),
    "Yellow": ([20, 50, 50], [35, 255, 255]),
}

fig, axes = plt.subplots(1, len(colors), figsize=(15, 5))

for ax, (color_name, (lower, upper)) in zip(axes, colors.items()):
    lower = np.array(lower)
    upper = np.array(upper)
    mask = cv2.inRange(img_hsv, lower, upper)
    ax.imshow(mask, cmap='gray')
    ax.set_title(color_name)
    ax.axis('off')

plt.tight_layout()
plt.show()

# Create green mask
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])
mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

# Apply mask to original image (color region)
green_region = cv2.bitwise_and(img, img, mask=mask_green)
green_region_rgb = cv2.cvtColor(green_region, cv2.COLOR_BGR2RGB)

# Display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_rgb)
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(mask_green, cmap='gray')
axes[1].set_title("Green Mask")
axes[1].axis('off')

axes[2].imshow(green_region_rgb)
axes[2].set_title("Green Region")
axes[2].axis('off')

plt.tight_layout()
plt.show()

# 3x3 kernel (for morphology)
kernel = np.ones((3, 3), np.uint8)

# Morphological opening removes small noise while preserving text
opened = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=1)

# Closing restores small gaps and smooths text regions
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

# Display
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(mask_green, cmap='gray')
axes[0].set_title("Original Mask")
axes[0].axis('off')

axes[1].imshow(opened, cmap='gray')
axes[1].set_title("Opened")
axes[1].axis('off')

axes[2].imshow(closed, cmap='gray')
axes[2].set_title("Closed")
axes[2].axis('off')

axes[3].imshow(cv2.dilate(closed, kernel, iterations=1), cmap='gray')
axes[3].set_title("Closed + Dilation")
axes[3].axis('off')

plt.tight_layout()
plt.show()