#HW 1
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
cv2.imwrite('down_img.png', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#HW 2
path = "../image/orig_img.png"
img = cv2.imread(path)
H, W, C = img.shape
print(H,W,C)

#zero padding
padding_size = 1
padded_img = np.zeros((H+2*padding_size, W+2*padding_size, C)) #초기화
padded_img = np.pad(img, pad_width=1) #padding
padded_img = padded_img[:,:,1:4] #필요없는 padded part 제거
padded_img = np.array(padded_img)
p_H, p_W, p_C = padded_img.shape

upsampled_img = np.zeros((2*p_H, p_W, p_C)) #height만 두배
#print(upsampled_img.shape) #964, 362, 3
#upsampled img에 값 원본 이미지 추가
for p_c in range(0, p_C):
    for p_w in range(0, p_W):
        for p_h in range(0, p_H):
            if p_h+3<p_H:
                # 원본 img픽셀값이 들어가는 경우
                upsampled_img[p_h * 2, p_w, p_c] = padded_img[p_h, p_w, p_c]

                #보간값이 들어가는 경우
                value = round((-0.5 * padded_img[p_h, p_w, p_c] + 1.5 * padded_img[
                p_h + 1, p_w, p_c] - 1.5 * padded_img[p_h + 2, p_w, p_c] + 0.5 * padded_img[
                                                         p_h + 3, p_w, p_c]) * 0.5 ** 3 + (
                                                                padded_img[p_h, p_w, p_c] - 2.5 * padded_img[
                                                            p_h + 1, p_w, p_c] + 2 * padded_img[
                                                                    p_h + 2, p_w, p_c] - 0.5 * padded_img[
                                                                    p_h + 3, p_w, p_c]) * 0.5 ** 2 + (
                                                                0.5 * padded_img[p_h + 2, p_w, p_c] - 0.5 *
                                                                padded_img[p_h, p_w, p_c]) * 0.5 + padded_img[
                                                        p_h + 1, p_w, p_c])
                upsampled_img[p_h*2+1, p_w, p_c] = value
                #print(p_h*2+1)


cv2.imshow('upsampled_img', upsampled_img.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


#bicubic interpolation - 세로로 보간값을 계산
new_img = np.zeros((2*p_H, 2*p_W, p_C)) #최종 img
#print("new_img",new_img.shape) #964,724,3

for p_c in range(0, p_C):
    for p_w in range(0, p_W):
        for p_h in range(0, 2*p_H):
            if p_w+3<p_W:
                new_img[p_h, p_w*2, p_c] = upsampled_img[p_h,p_w,p_c]

                # 보간값이 들어가는 경우
                value = round((-0.5 * upsampled_img[p_h, p_w, p_c] + 1.5 * upsampled_img[
                    p_h, p_w+1, p_c] - 1.5 * upsampled_img[p_h, p_w+2, p_c] + 0.5 * upsampled_img[
                                   p_h, p_w+3, p_c]) * 0.5 ** 3 + (
                                      upsampled_img[p_h, p_w, p_c] - 2.5 * upsampled_img[
                                  p_h, p_w+1, p_c] + 2 * upsampled_img[
                                          p_h, p_w+2, p_c] - 0.5 * upsampled_img[
                                          p_h, p_w+3, p_c]) * 0.5 ** 2 + (
                                      0.5 * upsampled_img[p_h, p_w+2, p_c] - 0.5 *
                                      upsampled_img[p_h, p_w, p_c]) * 0.5 + upsampled_img[
                                  p_h, p_w+1, p_c])
                new_img[p_h, p_w*2+1, p_c] = value

complete_img = np.zeros((2*H,2*W,C))
complete_img[:,:,:] = new_img[0:2*p_H-4,0:2*p_W-4,:]


cv2.imshow('new_img', complete_img.astype(np.uint8))
cv2.imwrite('up_img.png', complete_img)
cv2.waitKey(0)
cv2.destroyAllWindows()