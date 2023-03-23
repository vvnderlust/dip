import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib import cm #cm은 색상 맵을 나타내기 위함

path = "../image/orig_img.png"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
H, W = img.shape

kernel = np.array([[1,2,1],[2,4,2],[1,2,1]])
kernel_size = 3

new_img = np.zeros((H, W))

ww = W-kernel_size+1
hh = H-kernel_size+1

for w in range(ww):
    for h in range(hh):
        clipped_img = img[h:h+kernel_size, w:w+kernel_size]
        filtered_img = np.zeros((3, 3))
        filtered_img[:,:] = clipped_img[:,:] * kernel[:,:]
        value = np.sum(filtered_img) // 16
        value = round(value)
        new_img[h, w] = value

print(new_img.shape)
cv2.imshow('new_img', new_img.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()