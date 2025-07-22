#tkinter app with file dialog
import tkinter as tk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
from image_enhancement_pipeline import denoise, automatic_brightness_and_contrast, gaussian_blur, sharpen

root = tk.Tk()
root.title("Image Enhancement Pipeline")
root.geometry("400x200")

title_label = tk.Label(root, text="Image Enhancement Pipeline", font=("Arial", 16))
title_label.pack(pady=10)

select_button = tk.Button(root, text="Select Image", command=lambda: select_process_display())
select_button.pack(pady=10)

def process_image(image_matrix):
    denoised = denoise(image_matrix)
    contrast_adjusted, alpha, beta = automatic_brightness_and_contrast(denoised)
    blurred = gaussian_blur(contrast_adjusted)  
    sharpened = sharpen(blurred)
    return sharpened

def select_process_display():
    plt.close()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError("Image not found. Check the file path.")
    
    processed_image = process_image(image_matrix=image)
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Original Image")
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Processed Image")
    ax2.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()
    print(file_path)
    

root.mainloop()
