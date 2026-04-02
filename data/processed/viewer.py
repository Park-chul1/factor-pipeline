import numpy as np

arr = np.load("r.npy")

print(arr)           # 전체 출력
print(arr.shape)     # shape 확인
print(arr[:10])      # 일부만 보기
print(np.isnan(arr).sum())  # NaN 개수 확인