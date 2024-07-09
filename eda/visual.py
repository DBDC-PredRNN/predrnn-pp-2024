import numpy as np
import matplotlib.pyplot as plt
import os

# .npz 파일 로드
file_path = '/Users/songhune/Downloads/data/kth/test.npz'
data = np.load(file_path)

# 데이터 요약 함수
def summarize_data(data):
    for key in data.files:
        arr = data[key]
        print(f"{key} 요약:")
        print(f"형태: {arr.shape}")
        print(f"평균: {np.mean(arr)}")
        print(f"표준편차: {np.std(arr)}")
        print()

# 이미지 저장 함수
def save_frames(data, array_name, output_dir='/Users/songhune/Desktop/frames'):
    if array_name not in data.files:
        print(f"{array_name} 배열이 데이터셋에 없습니다.")
        return
    
    array_data = data[array_name]
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(array_data.shape[0]):
        fig, ax = plt.subplots()
        ax.imshow(array_data[i, 0], cmap='gray')
        ax.axis('off')
        temp_file = os.path.join(output_dir, f'frame_{i:04d}.png')
        plt.savefig(temp_file)
        plt.close(fig)
    print(f"이미지 저장 완료: {output_dir}")

# 각 배열의 이름과 shape 출력
summarize_data(data)

# input_raw_data 배열의 프레임을 이미지로 저장
save_frames(data, 'input_raw_data')
