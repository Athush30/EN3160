import cv2
import numpy as np
from matplotlib import pyplot as plt

def zoom_image(image, scale, interpolation):
    height, width = image.shape[:2]
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=interpolation)

def compute_normalized_ssd(img1, img2, bypass_size_error=True):
    if not bypass_size_error:
        assert img1.shape == img2.shape, "Images must be the same shape for SSD computation."
    else:
        min_height = min(img1.shape[0], img2.shape[0])
        min_width = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_height, :min_width]
        img2 = img2[:min_height, :min_width]
    ssd = np.sum((img1.astype("float32") - img2.astype("float32")) ** 2)
    norm_ssd = ssd / np.prod(img1.shape)
    return norm_ssd

def display_images(original, nearest, bilinear, titles):
    plt.figure(figsize=(15, 15))
    for idx, img in enumerate([original, nearest, bilinear]):
        plt.subplot(1, 3, idx+1)
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.title(titles[idx])
        plt.axis('off')
    plt.show()

def get_zoom_and_original_img(small_img, big_img, scale_factor=4, bypass_size_error=True):
    zoomed_nn = zoom_image(small_img, scale_factor, cv.INTER_NEAREST)
    zoomed_bilinear = zoom_image(small_img, scale_factor, cv.INTER_LINEAR)
    ssd_nn = compute_normalized_ssd(big_img, zoomed_nn, bypass_size_error=bypass_size_error)
    ssd_bilinear = compute_normalized_ssd(big_img, zoomed_bilinear, bypass_size_error=bypass_size_error)
    print(f"Normalized SSD (Nearest Neighbor): {ssd_nn}")
    print(f"Normalized SSD (Bilinear): {ssd_bilinear}")
    display_images(big_img, zoomed_nn, zoomed_bilinear, ["Original Image", "Nearest Neighbor Zoomed", "Bilinear Zoomed"])

small_img_1 = cv.imread('/content/im01small.png')
big_img_1   = cv.imread('/content/im01.png')
small_img_2 = cv.imread('/content/im02small.png')
big_img_2   = cv.imread('/content/im02.png')

get_zoom_and_original_img(small_img_1, big_img_1, scale_factor=4, bypass_size_error=False)
get_zoom_and_original_img(small_img_2, big_img_2, scale_factor=4, bypass_size_error=False)
