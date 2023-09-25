import numpy as np
import cv2
import matplotlib.pyplot as plt

def CannyFunction(img):
    img_copy = np.copy(img)
    image_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_copy,cv2.COLOR_RGB2GRAY)
    lower = 120
    upper = 240
    edge = cv2.Canny(gray, lower, upper)
    plt.subplot(2, 2, 1), plt.imshow(edge, 'gray'), plt.title('opencv_Canny')
    plt.show()