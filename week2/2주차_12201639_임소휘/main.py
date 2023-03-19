import numpy as np

from bgr2gray import bgr2gray
import cv2

path = "../cat.jpeg"
cat_img = cv2.imread(path) #3차원 행렬로 return
print(cat_img.shape)

Gray_img = bgr2gray(cat_img)

print(Gray_img.shape)
cv2.imshow('gray_img', Gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

new_img = np.zeros_like(Gray_img)

for i in range(Gray_img.shape[0]):
    for j in range(Gray_img.shape[1]):
        value = Gray_img[i, j]
        if value < 64:
            return_value = 64
        elif value < 128:
            return_value = 128
        elif value < 192:
            return_value = 192
        else:
            return_value = 255
        new_img[i, j] = return_value

cv2.imshow('gray_cat_img', new_img)
cv2.imwrite('gray_cat.png', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

