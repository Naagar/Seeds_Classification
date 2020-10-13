##  ____________  image_augmentation  ___________##


import numpy as np
import cv2
from matplotlib import pyplot as plt
from IPython.display import display, HTML

def show_img(img, figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(img)
    plt.imshow(img)
    

def show_augmentation(img, augmenation, **params):
    params_code = ', '.join(f'{key}={value}' for key, value in params.items())
    if params_code:
      params_code += ', '
    text = HTML(
        'Use this augmentation in your code:'
        '<pre style="display:block; background-color: #eee; margin: 10px; padding: 10px;">'
        f'{augmenation.__class__.__name__}({params_code}p=0.5)'
        '</pre>'
    )
    display(text)
    show_img(img)
    

image = cv2.imread('example.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (512, 512))

# display(HTML('<h3>Original image</h3>'))
show_img(image)

RandomGamma(gamma_limit=37, p=0.5)