import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./images/noisy_kojima.png') 
if image is None:
    raise FileNotFoundError("Image not found. Check the file path.")

def denoise(image):
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

def sharpen(image):
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    return cv2.filter2D(image, -1, sharpen_kernel)

def gaussian_blur(image, kernel_size=(3, 3)):
    return cv2.GaussianBlur(image, kernel_size, 0)

denoised = denoise(image)
contrast_adjusted, alpha, beta = automatic_brightness_and_contrast(denoised)
blurred = gaussian_blur(contrast_adjusted)  
sharpened = sharpen(blurred)

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title("Original Image")
ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title("Processed Image")
ax2.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()
