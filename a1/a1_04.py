import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#Import image
image = cv.imread('/content/spider.png')
assert image is not None

#converting image to hsv and extrating those values
image_hsv= cv.cvtColor(image,cv.COLOR_BGR2HSV)
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
h,s,v = cv.split(image_hsv)

# Create the figure for plotting
fig, ax = plt.subplots(1, 3, figsize=(12, 8))

ax[0].imshow(h, cmap='gray', vmin=0, vmax=255)
ax[0].set_title('Hue')
ax[0].axis("off")
ax[1].imshow(s, cmap='gray', vmin=0, vmax=255)
ax[1].set_title('Saturation')
ax[1].axis("off")
ax[2].imshow(v, cmap='gray', vmin=0, vmax=255)
ax[2].set_title('Value')
ax[2].axis("off")

plt.tight_layout()
plt.show()

#transformaion function
a=0.4
sigma=70
x=np.arange(0,256)
f=np.minimum(x+ a*128*np.exp(-((x-128)**2)/(2*sigma**2)),255).astype('uint8')

plt.figure(figsize=(5, 5))
plt.plot(x, f)
plt.title(f'Intensity Transformation f(x) with a = {a}')
plt.xlabel('Input Intensity (x)')
plt.ylabel('Output Intensity (f(x))')
plt.grid(True)
plt.xlim([0, 255])
plt.xlim([0, 255])
plt.show()

#apply transformation to s
s_modi=cv.LUT(s,f)

#Merge planes
merged = cv.merge([h,s_modi,v])
img_modified = cv.cvtColor(merged, cv.COLOR_HSV2RGB)

# Create a figure and axes
fig, axs = plt.subplots(1, 2, figsize=(12, 8))

# Plot the first image
axs[0].imshow(image_rgb)
axs[0].set_title('Original')
axs[0].axis('off')  # Turn off the axis

# Plot the second image
axs[1].imshow(img_modified)
axs[1].set_title('Vibrance adjusted')
axs[1].axis('off')  # Turn off the axis

# Show the plot
plt.tight_layout()
plt.show()
