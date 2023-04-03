import cv2
import numpy as np
from matplotlib import pyplot as plt

#흑백 이미지 불러오기
path = "./bean.jpg"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
H ,W = img.shape
#np.histogram 함수를 이용해 히스토그램 값 추출
#Tip: np.histogram()
hist, bins = np.histogram(img.ravel(), 256, [0,256])

#히스토그램의 cdf 계산 및 정규화
#Tip: np.cumsum()
cdf = np.cumsum(hist)
cdf_normalized = hist / cdf.max()

#Histogram Equalization(HE)
cdf_m = np.ma.masked_equal(cdf,0) #cdf에서 0인부분은 모두 mask처리
cdf_m = (cdf_m-cdf_m.min())*255 / ((H * W)-cdf_m.min())
cdf_scaled = np.ma.filled(cdf_m,0).astype('uint8')
HE_img = cdf_scaled[img]

hist_new = np.histogram(HE_img.ravel(), 256, [0,256])

#HE 적용 전/후 histogram 출력

plt.hist(img.ravel(), 256, [0,256])
plt.savefig('original img.png')
plt.show()

plt.hist(HE_img.ravel(), 256, [0,256])
plt.savefig('histogram equalized img.png')
plt.show()

#HE 적용 전/후 이미지 출력 및 저장
cv2.imshow('img', img)
cv2.waitKey()
cv2.imshow('HE result', HE_img)
cv2.waitKey()
cv2.imwrite('img.jpg', img)
cv2.imwrite('HE result.jpg', HE_img)
cv2.imwrite
cv2.destroyAllWindows()


