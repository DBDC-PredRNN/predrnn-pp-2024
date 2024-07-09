import os
import cv2
import numpy as np
from sklearn.model_selection import KFold

# 데이터셋 디렉토리 설정
data_dir = '/Users/songhune/Downloads/data/kth'
actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

# 비디오 파일을 읽어 numpy 배열로 변환
def read_videos(action):
    video_files = [os.path.join(data_dir, action, f) for f in os.listdir(os.path.join(data_dir, action)) if f.endswith('.avi')]
    clips = []
    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환
            frames.append(frame)
        cap.release()
        clips.append(np.array(frames))
    return clips

# 데이터셋 준비
clips = []
labels = []
for idx, action in enumerate(actions):
    action_clips = read_videos(action)
    clips.extend(action_clips)
    labels.extend([idx] * len(action_clips))

clips = np.array(clips)
labels = np.array(labels)

# K-Fold를 사용하여 데이터셋 분할
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_indices, test_indices = next(kf.split(clips))

train_clips = clips[train_indices]
train_labels = labels[train_indices]
test_clips = clips[test_indices]
test_labels = labels[test_indices]

# npz 파일로 저장
np.savez('training.npz', clips=train_clips, labels=train_labels)
np.savez('test.npz', clips=test_clips, labels=test_labels)

print('Data has been successfully split and saved to training.npz and test.npz.')
