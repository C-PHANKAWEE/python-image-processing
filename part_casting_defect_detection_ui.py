import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from PIL import Image, ImageTk
import cv2
import numpy as np
from skimage import transform

tf.debugging.set_log_device_placement(True)
model = tf.keras.models.load_model('./models/my_model.h5')

root = tk.Tk()
root.title("Image Enhancement Pipeline")
root.geometry("400x500")

title_label = tk.Label(root, text="Image Enhancement Pipeline", font=("Arial", 16))
title_label.pack(pady=10)

img = ImageTk.PhotoImage(file='./images/select_image_prompt.drawio.png')
imgLabel = tk.Label(root, image=img, height=300, width=300)

imgLabel.pack(pady=10)

def load_image(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (224, 224, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

def select_process_display():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    print(file_path)
    img = ImageTk.PhotoImage(file=file_path)
    imgLabel.config(image=img)
    imgLabel.image = img
    if file_path:
        image = load_image(file_path)
        pred = model.predict(image)
        isOk = np.argmax(pred)
        if isOk:
            result_label.config(text="OK", fg='green')
        else :
            result_label.config(text="Defect", fg='red')
        result_label.pack()
        if image is None:
            raise FileNotFoundError("Image not found. Check the file path.")
        
select_button = tk.Button(root, text="Select Image", command=lambda: select_process_display())
select_button.pack(pady=10)

result_label = tk.Label(root, font=("Arial", 12))
root.mainloop()