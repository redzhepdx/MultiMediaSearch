import argparse
import json
import os.path
from functools import partial
from multiprocessing import Pool
from typing import Dict, List

from pytube import Search, YouTube
from pytube.exceptions import AgeRestrictedError
from tqdm import tqdm


def download_video(yt_stream: YouTube, output_path: str, output_name: str) -> bool:
    """

    :param yt_stream:
    :param output_path:
    :param output_name:
    :return:
    """
    try:
        video = yt_stream.streams.get_by_resolution("480") or yt_stream.streams.get_by_resolution("360p")
        if not video:
            return False
        video.download(output_path=output_path, filename=output_name)
    except AgeRestrictedError:
        print("Age restricted video, skipping")
        return False
    except KeyError:
        print("Key error, skipping")
        return False
    return True


def get_videos_from_youtube(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    :param query:
    :param max_results:
    :return:
    """
    results = Search(query).results
    videos = []
    for result in results[:max_results]:
        video = {
            "title": result.title,
            "url": result.watch_url,
            "channel_url": result.channel_url,
            "views": result.views,
            "duration": result.length,
            "thumbnail": result.thumbnail_url,
        }
        if video["duration"] > 300:
            continue
        videos.append(video)
    return videos


def search_and_download_video(dataset_path: str, game_id: str, query: str) -> bool:
    videos = get_videos_from_youtube(query)
    video_retrieved = False
    for video in videos:
        url = video["url"]
        stream = YouTube(url, use_oauth=True)
        download_res = download_video(stream, os.path.join(dataset_path, "trailers"), f"{game_id}.mp4")
        video_retrieved |= download_res
        if download_res:
            break
    return video_retrieved


def process_annotation_file(annotation_file: str, dataset_path: str) -> None:
    annotation = json.load(open(os.path.join(dataset_path, "annotations", annotation_file)))
    video_path = os.path.join(dataset_path, f"trailers/{annotation['product_id']}.mp4")
    if os.path.exists(video_path):
        return
    game_name = annotation["product_name"]
    game_id = annotation["product_id"]
    query = f"{game_name} game trailer"
    video_retrieved = search_and_download_video(dataset_path, game_id, query)

    if not video_retrieved:
        print(f"Failed to retrieve video for {game_name}-{game_id}, trying gameplay trailer keyword")
        query = f"{game_name} gameplay trailer"
        video_retrieved = search_and_download_video(dataset_path, game_id, query)

    if not video_retrieved:
        print(f"Failed to retrieve video for {game_name}-{game_id}")


def get_trailer_videos(dataset_path: str) -> None:
    annotation_files = os.listdir(os.path.join(dataset_path, "annotations"))
    num_processes = os.cpu_count()

    with Pool(num_processes) as pool:
        partial_process = partial(process_annotation_file, dataset_path=dataset_path)
        list(tqdm(pool.imap(partial_process, annotation_files), total=len(annotation_files)))


def get_args():
    parser = argparse.ArgumentParser(description="Download videos from YouTube")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    return parser.parse_args()


def main():
    args = get_args()
    dataset_path = args.dataset_path
    videos_path = os.path.join(dataset_path, "trailers")
    os.makedirs(videos_path, exist_ok=True)
    get_trailer_videos(dataset_path)


if __name__ == "__main__":
    main()
