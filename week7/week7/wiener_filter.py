import cv2
import matplotlib.pyplot as plt
import numpy as np

def ideal_BRF(image):
    M, N = image.shape
    H = np.ones((M, N))
    U0 = int(M / 2)
    V0 = int(N / 2)
    D0 = 200  # cutoff
    W = 5  # W=width

    for u in range(M):
        for v in range(N):
            if (D0 - W) <= np.sqrt((u - U0) ** 2 + (v - V0) ** 2) <= (D0 + W):
                H[u, v] = 0
    return H


def wiener_filter(H, image):
    M, N = image.shape
    W = np.zeros_like(image)
    K = 0.5

    for u in range(M):
        for v in range(N):
            H_2 = np.abs(H[u, v]) ** 2
            W[u, v] = H_2 / (H[u, v] * (H_2 + K))

    print(W[300, 300])
    return W


if __name__ == "__main__":
    image = cv2.imread('noisy.jpg', 0)
    #plt.imshow(np.uint8(image), cmap='gray'), plt.axis('off')
    #plt.show()

    M, N = image.shape

    # Zero padding
    padded_image = np.zeros((2 * M, 2 * N))
    padded_image[:M, :N] = image

    # FFT
    fft_image = np.fft.fftshift(np.fft.fft2(padded_image))
    fft_image_ = np.log(np.abs(fft_image))
    #plt.imshow(fft_image_.real.astype(np.uint8), cmap='gray')
    #plt.show()

    # wiener_filter 계산
    sigma = 1e-5
    H = 1 / (ideal_BRF(fft_image) + sigma)

    w_filter = wiener_filter(H, fft_image)

    # Filtering
    filtered_image = np.multiply(fft_image, w_filter)

    # IDFT
    idft_image = np.fft.ifft2(np.fft.fftshift(filtered_image)).real
    # Remove padded part
    idft_image = idft_image[:M, :N]

    cv2.imshow('idft_image', idft_image.astype(np.uint8))
    cv2.imwrite('k_0.05.jpg', idft_image.astype(np.uint8))
    cv2.waitKey(0)

