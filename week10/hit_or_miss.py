import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_img():
    img = np.zeros((7, 15))
    for i in range(7):
        for j in range(15):
            if i == 1 or i == 5:
                if j >= 1 and j <= 4:
                    img[i, j] = 1
                if j >= 10 and j <= 13:
                    img[i, j] = 1
            if i == 2 or i == 4:
                if j >=1 and j <= 13:
                    img[i, j] = 1
            if i == 3:
                if j >=1 and j <= 13:
                    img[i, j] = 1
                img[i, 5] = 0
                img[i, 9] = 0
    img = img.astype(np.uint8)
    return img

###########################
# Implement this function #
###########################
import numpy as np
def hit_or_miss(img, kernel):
    M, N = img.shape
    k_size, _ = kernel.shape

    pad = k_size // 2
    img_padded = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=0)

    result1 = np.zeros((M, N))
    result2 = np.zeros((M, N))

    # Erosion 1
    for i in range(M):
        for j in range(N):
            boundary = img_padded[i:i+k_size, j:j+k_size] * kernel
            if np.min(boundary) == 0:
                result1[i, j] = 255
            else:
                result1[i, j] = 0

    # Erosion 2
    kernel_inv = -1 * kernel
    img_c = 1 - img_padded

    for i in range(M):
        for j in range(N):
            boundary = img_c[i:i + k_size, j:j + k_size] * kernel_inv
            if np.min(boundary) == 0:
                result2[i, j] = 255
            else:
                result2[i, j] = 0

    # Hit or Miss Transform
    result = np.zeros((M, N))

    for i in range(M):
        for j in range(N):
            if result1[i, j] == 255 and result2[i, j] == 255:
                result[i, j] = 255

    return result


if __name__ == "__main__":    
    img = get_img()
    plt.imshow(img, cmap="gray"), plt.show()
    
    B = np.array([[1, 1, 1],
                  [1, -1, 1],
                  [1, 1, 1]], dtype="int")
    plt.imshow(B, cmap="gray"), plt.show()

    cv_result = cv2.morphologyEx(img, cv2.MORPH_HITMISS, B)
    my_result = hit_or_miss(img, B)
    
    plt.imshow(cv_result, cmap="gray"), plt.show()
    plt.imshow(my_result, cmap="gray"), plt.show()
    