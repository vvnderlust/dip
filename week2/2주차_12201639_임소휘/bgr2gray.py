import cv2
import numpy as np
def bgr2gray(img):
    b, g, r = cv2.split(img)
    b = b * 0.114
    g = g * 0.587
    r = r * 0.299
    b = b.astype(np.uint8)
    g = g.astype(np.uint8)
    r = r.astype(np.uint8)
    img_gray = b + g + r

    return img_gray




