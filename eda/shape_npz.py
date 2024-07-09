import numpy as np

# .npz 파일 로드
file_path = '/Users/songhune/Library/Mobile Documents/com~apple~CloudDocs/Workspace/moving-mnist-train.npz'
data = np.load(file_path)

# 각 배열의 이름과 shape 출력
for key in data.files:
    print(f"Array name: {key}")
    print(f"Array shape: {data[key].shape}")
    print()

# 예시 결과 출력
# Array name: train
# Array shape: (10000, 20, 64, 64)
# 
# Array name: test
# Array shape: (2000, 20, 64, 64)