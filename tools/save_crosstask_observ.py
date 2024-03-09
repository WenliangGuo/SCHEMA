import os
import numpy as np
import shutil
import cv2

# load annotations
from utils import get_all_task_videos, get_video_path, parse_annotation, parse_task_info

task_vids = get_all_task_videos()
task_info = parse_task_info('../dataset/crosstask/crosstask_release/tasks_primary.txt')
anot_info, action_collect = parse_annotation('../dataset/crosstask/crosstask_release/annotations', task_info)

image_save_dir = '~/data/cross_task_images'
if not os.path.exists(image_save_dir):
    os.makedirs(image_save_dir)

for idx, (task, vid) in enumerate(task_vids):
    is_exist, video_path = get_video_path(vid)
    if not is_exist:
        print(f'{vid} video not exist')
        continue

    save_dir = os.path.join(image_save_dir, vid)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        continue
    # load video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    count = 0
    images = []
    sampled_frame_count = int(np.round(frame_count / fps))
    for i in range(0, sampled_frame_count, 1):
        # Set frame index
        index = min(int(fps * i), frame_count-1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        # Read frame
        frame_exist, frame = cap.read()
        if frame_exist == False:
            break
        count += 1

        # convert frames
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))
        images.append(frame)

    if len(images) == 0:
        shutil.rmtree(save_dir)
        print(f"{idx}: error, vid: {vid}")
        continue
    
    images = np.stack(images)
    cnt = 0

    try:
        for action_info in anot_info[vid]:
            start, end = action_info['start']-1, action_info['end']-1
            s_offset_start = max(0, start-1)
            s_offset_end = min(start+2,images.shape[0])
            e_offset_start = max(0, end-1)
            e_offset_end = min(end+2,images.shape[0])

            obser_start_smooth = np.mean(images[s_offset_start:s_offset_end], axis = 0)
            obser_end_smooth = np.mean(images[e_offset_start:e_offset_end], axis = 0)

            obser_start = images[start]
            obser_end = images[end]

            cv2.imwrite(os.path.join(save_dir, f'{cnt}_0_smooth.png'), obser_start_smooth)
            cv2.imwrite(os.path.join(save_dir, f'{cnt}_0.png'), obser_start)
            cv2.imwrite(os.path.join(save_dir, f'{cnt}_1_smooth.png'), obser_end_smooth)
            cv2.imwrite(os.path.join(save_dir, f'{cnt}_1.png'), obser_end)
            cnt += 1
        print("saved video: ", vid)

    except:
        print(f"error: {vid}", images.shape)
        shutil.rmtree(save_dir)
    
cap.release()