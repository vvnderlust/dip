import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
    img_c = 255 - img_padded

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

def thinning(img, kernel_list, r):
    for i in range(r):
        for kernel in kernel_list:
            hmt_AB = hit_or_miss(img, kernel)
            hmt_AB_c = np.where(hmt_AB > 0, 0, 255)
            img = np.where((img > 0) & (hmt_AB_c > 0), 255, 0).astype(np.uint8)
        plt.imshow(img, cmap="gray"), plt.show()
    return img


def getimg_thinning():
    img = np.zeros((5,11))
    img[0, :] = 255
    img[1:4, :9] = 255
    img[4, :3] = 255
    img[4, 5:9] = 255

    return img.astype(np.uint8)


if __name__ == "__main__":
    img = getimg_thinning()
    plt.imshow(img, cmap="gray"), plt.show()



    B1 = np.array([[-1, -1, -1],
                   [0, 1, 0],
                   [1, 1, 1]], dtype="int")
    B2 = np.array([[0, -1, -1],
                   [1, 1, -1],
                   [1, 1, 0]], dtype="int")
    B3 = np.rot90(B1, 3)
    B5 = np.rot90(B3, 3)
    B7 = np.rot90(B5, 3)
    B4 = np.rot90(B2, 3)
    B6 = np.rot90(B4, 3)
    B8 = np.rot90(B6, 3)

    thin_kernel_list = [B1, B2, B3, B4, B5, B6, B7, B8]

    thin_img = thinning(img, thin_kernel_list, 2)

    plt.imshow(thin_img, cmap='gray'), plt.show()

    thin_kernel_list.reverse()

    ####thickening####
    img_c = 255-thin_img
    thick_img = thinning(img_c, thin_kernel_list, 1) #A_c에 thinning 취함

    thick_img = 255 - thick_img
    plt.imshow(thick_img, cmap='gray'), plt.show()