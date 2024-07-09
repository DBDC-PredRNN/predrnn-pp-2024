import imageio
import os

# GIF 생성 함수
def create_gif_from_frames(frame_dir='/Users/songhune/Desktop/frames', gif_name='/Users/songhune/Desktop/frames/animation.gif', fps=5):
    frames = []
    for frame_file in sorted(os.listdir(frame_dir)):
        if frame_file.endswith('.png'):
            frame_path = os.path.join(frame_dir, frame_file)
            frames.append(imageio.imread(frame_path))
    
    imageio.mimsave(gif_name, frames, fps=fps)
    print(f"GIF 생성 완료: {gif_name}")

# 저장된 프레임을 사용하여 GIF 생성
create_gif_from_frames()
