import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
#import image
image = cv.imread('/content/jeniffer.jpg')
assert image is not None

#convet and sl=plit hsv
image_hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)
h,s,v = cv.split(image_hsv)

image_rgb=cv.cvtColor(image,cv.COLOR_BGR2RGB)

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

#apply binary masking in s plane
_, mask_s = cv.threshold(s, 12, 255, cv.THRESH_BINARY)


#extracting foreground using mask
foreground = cv.bitwise_and(image, image, mask=mask_s)

# Compute and plot the histogram of the Value (V) channel of the foreground
foreground_hsv = cv.cvtColor(foreground, cv.COLOR_BGR2HSV)
H_foreground, S_foreground, V_foreground = cv.split(foreground_hsv)


# Calculate the histogram 
hist = cv.calcHist([V_foreground], [0], mask_s, [256], [0, 256])
x_positions = np.arange(len(hist))

#calculate cumulative sum
cdf = hist.cumsum()

# Number of pixels
pixels = cdf[-1]

# Define transformation
transformation = np.array([(256-1)/(pixels)*cdf[k] for k in range(256)]).astype("uint8")

# Equalize
eq = transformation[V_foreground]

# Calculate the histogram of the equalized Value channel
hist1 = cv.calcHist([eq], [0], mask_s, [256], [0, 256])

# Create an array for the x positions of the bars
x1_positions = np.arange(len(hist1))

# Merge
merged = cv.merge([H_foreground, S_foreground, V_foreground])
foreground_modified = cv.cvtColor(merged, cv.COLOR_HSV2RGB)

# Extract the background
background = cv.bitwise_and(image, image, mask=cv.bitwise_not(mask_s))

# Merge the equalized foreground and background
result = cv.add(cv.cvtColor(background, cv.COLOR_BGR2RGB), foreground_modified)

plt.figure()
plt.imshow(cv.cvtColor(mask_s, cv.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')
plt.show()

# Plot the cdf
plt.figure()
plt.plot(cdf, color='black')
plt.title('Cumulative Sum of the Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot the histogram as a bar plot
plt.figure()
plt.bar(x_positions, hist.flatten(), color='black', width=1)  # Use width=1 for each bar
plt.title('Histogram of Value Channel for Foreground')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])  # Set x-axis limits
plt.grid(True)
plt.show()

plt.figure()
plt.bar(x1_positions, hist1.flatten(), color='black', width=1)  # Use width=1 for each bar
plt.title('Equalized Histogram of Value Channel for Foreground')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])  # Set x-axis limits
plt.grid(True)
plt.show()

# Display the eq_foreground
plt.figure(figsize=(10, 5))
plt.subplot(1,3,1)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(cv.cvtColor(foreground, cv.COLOR_BGR2RGB))
plt.title('Foreground')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(result)
plt.title('Foreground equalized')
plt.axis('off')
plt.tight_layout()
plt.show()
