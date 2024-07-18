import os
import cv2
import numpy as np

def process_video_file(filepath, frame_width, frame_height):
    cap = cv2.VideoCapture(filepath)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (frame_width, frame_height))
        frames.append(resized_frame)
    cap.release()
    return frames

def create_clips_and_data_from_videos(input_folder, person_splits, frame_width, frame_height):
    clips = []
    input_raw_data = []
    current_index = 0
    person_ids = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".avi"):
            person_id = filename.split('_')[0].replace('person', '')
            if person_id in person_splits:
                filepath = os.path.join(input_folder, filename)
                frames = process_video_file(filepath, frame_width, frame_height)

                num_frames = len(frames)
                seq_length = num_frames // 4
                for start in range(0, num_frames - seq_length * 2, seq_length):
                    input_clip = frames[start:start + seq_length]
                    output_clip = frames[start + seq_length:start + seq_length * 2]

                    clips.append([current_index, seq_length])
                    input_raw_data.extend(input_clip)
                    current_index += seq_length

                    clips.append([current_index, seq_length])
                    input_raw_data.extend(output_clip)
                    current_index += seq_length

                    person_ids.append(person_id)
                    person_ids.append(person_id)

    dims = np.array([[1, frame_height, frame_width]], dtype=np.int32)
    input_raw_data = np.array(input_raw_data, dtype=np.float32).reshape(-1, 1, frame_height, frame_width)
    return clips, dims, input_raw_data, person_ids

def split_dataset_personwise(clips, input_raw_data, person_ids, person_splits):
    splits = {'train': [], 'validation': [], 'test': []}
    for split_type in ['train', 'validation', 'test']:
        split_indices = [i for i, person_id in enumerate(person_ids) if person_splits[person_id] == split_type]
        split_clips = np.array([clips[i] for i in split_indices])
        split_data_indices = []
        for ind in split_indices:
            split_data_indices.extend(range(int(clips[ind][0]), int(clips[ind][0]) + int(clips[ind][1])))
        split_data = input_raw_data[split_data_indices]
        splits[split_type].append((split_clips, split_data))
    return splits

def save_datasets_personwise(splits, dims, output_path):
    for split_type, split_data in splits.items():
        split_output_path = os.path.join(output_path, split_type)
        os.makedirs(split_output_path, exist_ok=True)
        for i, (clips, data) in enumerate(split_data):
            np.savez(os.path.join(split_output_path, f'{split_type}_dataset_{i+1}.npz'), clips=clips, dims=dims, input_raw_data=data)

input_folder = '/Users/songhune/Downloads/data/kth'
output_path = '/Users/songhune/Downloads/data/kth_preprocessed'
frame_width = 160
frame_height = 120

person_splits = {
    '11': 'train', '12': 'train', '13': 'train', '14': 'train',
    '19': 'validation', '20': 'validation', '21': 'validation', '23': 'validation',
    '22': 'test', '02': 'test', '03': 'test', '05': 'test'
}

clips, dims, input_raw_data, person_ids = create_clips_and_data_from_videos(input_folder, person_splits, frame_width, frame_height)
splits = split_dataset_personwise(clips, input_raw_data, person_ids, person_splits)
save_datasets_personwise(splits, dims, output_path)
