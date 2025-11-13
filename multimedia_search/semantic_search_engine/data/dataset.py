import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import av
import cv2
import imutils
import natsort
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, VideoMAEImageProcessor
from transformers.models.auto.image_processing_auto import AutoImageProcessor


def sample_frame_indices(clip_len: int, frame_sample_rate: float, seg_len: int) -> np.ndarray:
    """
    Adapted from https://huggingface.co/docs/transformers/model_doc/videomae
    :param clip_len:
    :param frame_sample_rate:
    :param seg_len:
    :return:
    """
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def read_video_pyav(container: av.container.InputContainer, indices: np.ndarray) -> np.ndarray:
    """
    Adapted from https://huggingface.co/docs/transformers/model_doc/videomae
    :param container:
    :param indices:
    :return:
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


class MultiMediaDataset(Dataset):
    """

    :param images_folder:
    :param annotations_folder:
    :param text_tokenizer:
    :param img_preprocessor:
    :param videos_folder:
    :param video_processor:
    :param max_length:
    """

    def __init__(self,
                 images_folder: str,
                 annotations_folder: str,
                 text_tokenizer: DistilBertTokenizer,
                 img_preprocessor: Optional[A.Compose] = None,
                 videos_folder: Optional[str] = None,
                 video_processor: Optional[Union[AutoImageProcessor, VideoMAEImageProcessor]] = None,
                 max_length: int = 512,
                 use_images: bool = True,
                 use_videos: bool = True) -> None:

        self.images_folder = images_folder
        self.annotations_folder = annotations_folder
        self.img_preprocessor = img_preprocessor
        self.text_tokenizer = text_tokenizer

        self.use_images = use_images
        self.use_videos = use_videos

        self.video_processor = video_processor
        self.videos_folder = videos_folder

        self.image_paths = natsort.natsorted(os.listdir(self.images_folder))
        self.summaries, summary_names, none_summaries = self.read_all_summaries()

        if self.videos_folder is not None and self.use_videos:
            self.video_paths = natsort.natsorted(os.listdir(self.videos_folder))

        for none_summary in none_summaries:
            # the image extension is not always jpg, it can be png or gif
            for image_path in self.image_paths:
                if image_path.startswith(none_summary.replace(".json", ".")):
                    self.image_paths.remove(image_path)
                    break

            if self.videos_folder is not None and self.use_videos:
                self.video_paths.remove(none_summary.replace(".json", ".mp4"))

        assert len(self.image_paths) == len(self.summaries)

        # One more alignment check
        for image_path, summary_path in zip(self.image_paths, summary_names):
            if image_path.split(".")[0] != summary_path.split(".")[0]:
                print(f"Image path : {image_path}")
                print(f"Summary path : {summary_path}")
                raise ValueError("Image and summary paths do not match")

        print(f"Len Summaries : {len(self.summaries)}\n"
              f"Len Images : {len(self.image_paths)}")

        self.encoded_summaries = self.text_tokenizer(self.summaries, padding=True, truncation=True,
                                                     max_length=max_length)

    def read_all_summaries(self) -> Tuple[List[str], List[str], List[str]]:
        """

        :return:
        """
        summaries = []
        summary_names = []
        none_summary_names = []
        for annotation_path in natsort.natsorted(os.listdir(self.annotations_folder)):
            annotation = json.load(open(os.path.join(self.annotations_folder, annotation_path)))
            if annotation["product_summary"] is None:
                none_summary_names.append(annotation_path)
            else:
                summary_names.append(annotation_path)
                summaries.append(annotation["product_summary"])

        return summaries, summary_names, none_summary_names

    def __len__(self):
        """

        :return:
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """

        :param idx:
        :return:
        """
        item = {
                key: torch.tensor(values[idx])
                for key, values in self.encoded_summaries.items()
        }
        if self.use_images:
            image = cv2.imread(os.path.join(self.images_folder, self.image_paths[idx]))
            if image is None:
                # Create a black image
                image = np.zeros((512, 512, 3), dtype=np.uint8)
                print(f"BROKEN IMAGE: {self.image_paths[idx]}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.img_preprocessor(image=image)["image"]
            item["image"] = torch.tensor(image).permute(2, 0, 1).float()
        item["summary"] = self.summaries[idx]

        if self.use_videos:
            stream_path = os.path.join(self.videos_folder, self.video_paths[idx])
            if os.path.isfile(stream_path):
                container = av.open(os.path.join(self.videos_folder, self.video_paths[idx]))
                indices = sample_frame_indices(clip_len=32, frame_sample_rate=0.5, seg_len=100)
                stream_data = read_video_pyav(container, indices)
            elif os.path.isdir(stream_path):
                frames = []
                for frame_path in natsort.natsorted(os.listdir(stream_path)):
                    frame = cv2.imread(os.path.join(stream_path, frame_path))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = imutils.resize(frame, width=224, height=224)
                    frames.append(frame)
                stream_data = np.zeros((16, 224, 224, 3), dtype=np.uint8)
                stream_data[:len(frames)] = frames
            else:
                raise ValueError(f"Stream path {stream_path} is neither a file nor a directory")
            video = self.video_processor(stream_data, return_tensors="pt")
            item["video"] = video

        return item
