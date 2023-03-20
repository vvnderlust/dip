import numpy as np
import cv2

n= np.array([1,2])
print(n.shape)
#list끼리 더하고 합치는 것은 원소별 연산이 아니기 때문에 numpy.array를 사용해야 함
# numpy를 사용하면 차원만 동일하면 성분 별 연산이 가능함

#openCV: 영상처리를 위해 명령어를 제공, image 활용이 가능한 라이브러리.

print(cv2.__version__)