import json
import os
from functools import partial
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import natsort
import requests
import tqdm
import yaml


def create_dataset(data_file_path: str, output_folder: str) -> None:
    """

    :param data_file_path:
    :param output_folder:
    :return:
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "annotations"), exist_ok=True)

    data = json.load(open(data_file_path))["data"]

    num_processes = os.cpu_count()
    with Pool(num_processes) as pool:
        create_product_annotation_fn = partial(create_product_annotation, output_folder=output_folder)
        data_with_ids = [(product, product_id) for product_id, product in enumerate(data)]
        list(tqdm.tqdm(pool.imap(create_product_annotation_fn, data_with_ids), total=len(data)))


def create_product_annotation(product_with_id: Tuple[Dict[str, Any], int], output_folder: str) -> None:
    """

    :param product_with_id:
    :param output_folder:
    :return:
    """
    product = product_with_id[0]
    product_id = product_with_id[1]

    product_name = product["name"]
    product_summary = product["summary"]
    platforms = product["platforms"]
    genres = product["genres"]
    release_dates = product["releaseDates"]
    themes = product.get("themes", [])
    modes = product.get("modes", [])
    image_url = product["coverURL"]

    last_release_date = max(release_dates) if len(release_dates) != 0 else None
    response = requests.get(image_url)
    if response is None or response.status_code != 200:
        print(f"No image found : {product_name}")
        return

    ext = response.headers.get("content-type").split("/")[-1]
    # Retrieve image and save it to the images folder
    image = response.content

    # check if there is an image
    if image is None:
        print(f"Image is none : {product_name}")
        return

    with open(os.path.join(output_folder, "images", f"{product_id}.{ext}"), "wb") as f:
        f.write(image)

    annotation = {
        "product_id": product_id,
        "product_name": product_name,
        "product_summary": product_summary,
        "platforms": platforms,
        "genres": genres,
        "last_release_date": last_release_date,
        "themes": themes,
        "modes": modes
    }
    with open(os.path.join(output_folder, "annotations", f"{product_id}.json"), "w") as f:
        json.dump(annotation, f, indent=4)


def get_metadata(folder_path: str) -> List[Dict[str, Any]]:
    """

    :param folder_path:
    :return:
    """
    annotations_folder = os.path.join(folder_path, "annotations")
    annotations_files = natsort.natsorted(os.listdir(annotations_folder))
    annotations = []

    for annotation_file in annotations_files:
        annotation = yaml.load(open(os.path.join(annotations_folder, annotation_file)), Loader=yaml.FullLoader)
        if annotation["product_summary"] is None:
            continue
        if annotation.get("themes"):
            annotation["themes"] = [theme['id'] for theme in annotation["themes"]]
        annotations.append(annotation)

    return annotations


def main():
    create_dataset(data_file_path="data.json", output_folder="gamegator_dataset_v2")


if __name__ == "__main__":
    main()
