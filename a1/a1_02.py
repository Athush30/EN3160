import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#white matter
#importing image
image = cv.imread('/content/Screenshot 2025-08-12 071150.png', cv.IMREAD_GRAYSCALE)
assert image is not None

#gaussian pulse for white matter
mu=155
sigma=20
x = np.linspace(0, 255, 256)
gaussian_white = 255 * np.exp(-((x - mu)**2) / (2 * sigma**2))

# Ensure the output is in the valid range for image intensities [0, 255]
gaussian_white = np.clip(gaussian_white, 0, 255)

print(gaussian_white.shape)

#plot the array
plt.figure(figsize=(5, 5))
plt.plot(gaussian_white)
plt.xlabel("Input intensity")
plt.xlim(0, 255)
plt.ylim(0, 255)
plt.ylabel("Output intensity")
plt.grid(True)
plt.show()

g = gaussian_white[image]

# Display the image
plt.figure(figsize=(5, 5))
plt.imshow(g, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()




#Grey matter
#importing image
image = cv.imread('/content/Screenshot 2025-08-12 071150.png', cv.IMREAD_GRAYSCALE)
assert image is not None

#gaussian pulse for white matter
mu=200
sigma=10
x = np.linspace(0, 255, 256)
t = 255 * np.exp(-((x - mu)**2) / (2 * sigma**2))

# Ensure the output is in the valid range for image intensities [0, 255]
t = np.clip(t, 0, 255)

print(t.shape)

#plot the array
plt.figure(figsize=(5, 5))
plt.plot(t)
plt.xlabel("Input intensity")
plt.xlim(0, 255)
plt.ylim(0, 255)
plt.ylabel("Output intensity")
plt.grid(True)
plt.show()

g = t[image]

# Display the image
plt.figure(figsize=(5, 5))
plt.imshow(g, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()

