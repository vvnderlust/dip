import cv2
import numpy as np
import matplotlib.pyplot as plt
def DFT(padded_image):
    M, N = padded_image.shape
    dft2d = np.zeros((M, N), dtype=complex)

    for k in range(M):
        for l in range(N):
            sum_ = 0.0
            for m in range(M):
                for n in range(N):
                    e = np.exp(-2j * np.pi * ((k * m) / M + (l * n) / N))
                    sum_ += padded_image[m, n] * e
            dft2d[k, l] = sum_
    return dft2d

def IDFT(dft_image):
    M, N = dft_image.shape
    idft2d = np.zeros((M,N), dtype=complex)

    for k in range(M):
        for l in range(N):
            sum_ = 0.0
            for m in range(M):
                for n in range(N):
                    e = np.exp(2j * np.pi * ((k * m) / M + (l * n) / N))
                    sum_ += dft_image[m, n] * e
            idft2d[k, l] = sum_
    return idft2d

image = cv2.imread('../image/cat.jpeg', 0)  # 1은 color, 0은 gray scale을 의미
image = cv2.resize(image, (50, 50))
plt.imshow(image, cmap='gray')
plt.show()

M, N = image.shape

# zero padding
P, Q = 2 * M, 2 * N
padded_image = np.zeros((P, Q))
padded_image[:M, :N] = image
plt.imshow(padded_image, cmap='gray')
plt.show()

# Centering
padded_image_new = np.zeros((P, Q))
for x in range(P):
    for y in range(Q):
        padded_image_new[x, y] = padded_image[x, y] * ((-1) ** (x + y))

# DFT
dft2d = DFT(padded_image_new)
print('Complete the DFT with centering')
plt.imshow(dft2d.real, cmap='gray')
plt.savefig('dft2d_img.png')
plt.show()


#Inverse DFT
idft2d = IDFT(dft2d)

#De-centering
for x in range(P):
    for y in range(Q):
        idft2d[x,y] = idft2d[x,y] * ((-1) ** (x+y))

#Remove zero-padding part
idft2d = idft2d[:int(P/2), :int(Q/2)].real
plt.imshow(idft2d, cmap='gray')
plt.savefig('idft2d_img.png')
plt.show()
