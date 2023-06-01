import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = '../image/lena.png'
img = cv2.imread(img_path, 1)

# RGB 영역에서 color smoothing 수행
filter = np.ones((3, 3), dtype=np.float32) / 9

filtered_rgb = cv2.filter2D(img, -1, filter)

# HSI 영역에서 color smoothing 수행
hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
i = hsi[:, :, 2]
filtered_hsi = cv2.filter2D(i, -1, filter)
hsi[:, :, 2] = filtered_hsi
hsi_result = cv2.cvtColor(hsi, cv2.COLOR_HSV2BGR)

rgb_gray = cv2.cvtColor(filtered_rgb, cv2.COLOR_BGR2GRAY)
hsi_gray = cv2.cvtColor(hsi_result, cv2.COLOR_BGR2GRAY)

result = np.hstack((img, filtered_rgb, hsi_result))
cv2.imshow('result(left:original, center:rgb, right:hsi)', result)

cv2.waitKey(0)

diff = np.abs(hsi_gray.astype(np.int32) - rgb_gray.astype(np.int32)).astype(np.uint8)
diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

# 결과 이미지 출력
cv2.imshow('Difference', diff)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('DIfference.png', diff)
cv2.imwrite('rgb.png', filtered_rgb)
cv2.imwrite('hsi.png',hsi_result)
# 차이가 발생하는 이유는 rgb에서 smoothing한 경우에는 각 채널 별 smoothing을 수행하고 합쳤기 때문이다.
# 반면 HSI의 경우에는 intensity, 즉 rgb값의 평균을 낸 값에서 smoothing을 수행했기 때문이다.
