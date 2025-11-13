# create mock up videos in the data folder
import cv2
import numpy as np
import tqdm


def create_random_videos(videos_path: str, video_count: int, frame_count: int = 100) -> None:
    """

    :param videos_path:
    :param video_count:
    :param frame_count:
    :return:
    """
    for vid_id in tqdm.tqdm(range(video_count)):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(f"{videos_path}/{vid_id}.mp4", fourcc, 30, (640, 480))
        for _ in range(frame_count):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            video.write(frame)
        video.release()


def main():
    videos_path = "/home/redzhep/workspace_redzhep/personal_projects/MultiMediaSearch/gamegator_dataset_v1/videos"
    video_count = 4543
    create_random_videos(videos_path, video_count)


if __name__ == '__main__':
    main()
