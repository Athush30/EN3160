import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#importing image
image = cv.imread('/content/Screenshot 2025-08-12 063950.png', cv.IMREAD_GRAYSCALE)
assert image is not None

#piecewise transformation array
t1 = np.linspace(0,50,num=51).astype('uint8')
t2 = np.linspace(100,255,num=100).astype('uint8')
t3 = np.linspace(150,255,num=105).astype('uint8')

#concatenating all to create a transformation array
t=np.concatenate((t1,t2,t3), axis=0).astype('uint8')
print(t.shape)

# Display the array
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
plt.figure(figsize=(6, 6))
plt.imshow(g, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()
