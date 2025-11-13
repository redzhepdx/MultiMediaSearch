import argparse
import base64
import json
import math
import os
import time

import cv2
import numpy as np
import requests
from matplotlib import pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-so", "--search_option", choices=["text", "image", "game", "new_game"],
                        type=str, required=False, default="similarity")
    parser.add_argument("-st", "--search_text", type=str, required=False,
                        default="I want to play a story game and the story has to change based on my decisions.")
    parser.add_argument("-imp", "--image_path", type=str, required=False)
    parser.add_argument("-gid", "--game_id", type=int, required=False, default=0)
    parser.add_argument("-e", "--endpoint", type=str, required=False, default="gamegator-mm-search")
    parser.add_argument("-d", "--data_path", type=str, required=False,
                        default="MultiMediaSearch/gamegator_dataset_28k")
    return parser.parse_args()


def prepare_request(args):
    endpoint = args.endpoint
    game_id = args.game_id
    search_option = args.search_option
    api = f"http://localhost:8080/predictions/{endpoint}/"
    request_data = {}
    if search_option == "text":
        request_data["search_type"] = "text"
        request_data["search_input"] = args.search_text
    elif search_option == "image":
        image = cv2.imread(args.image_path)
        data = cv2.imencode(".jpg", image)[1].tobytes()
        im_b64 = base64.b64encode(data).decode("utf8")
        request_data = {"search_type": "image", "search_input": im_b64}
    elif search_option in ["game", "new_game"]:
        game_id = args.game_id
        request_data = {"search_type": search_option, "search_input": game_id}
        if search_option == "new_game":
            request_data = prepare_new_game_registry(args, game_id, request_data)
    return api, game_id, request_data


def prepare_new_game_registry(args, game_id, request_data):
    data_path = args.data_path
    random_image = None
    for ext in ["jpg", "jpeg", "png"]:
        random_image = cv2.imread(f"{data_path}/images/{game_id}.{ext}")
        if random_image is not None:
            break
    if random_image is not None:
        data = cv2.imencode(f".{ext}", random_image)[1].tobytes()
        game_data = json.load(open(os.path.join(data_path, "annotations", f"{game_id}.json"), "r"))
        game_data["product_id"] = str(int(game_data["product_id"]) + 25000)
        game_data["product_name"] = f"{game_data['product_name']} V2}}"
        game_data["summary"] = game_data["product_summary"]
        game_data["themes"] = [theme['id'] for theme in game_data["themes"]]
        game_data.pop("product_summary")
        im_b64 = base64.b64encode(data).decode("utf8")
        game_data["image"] = im_b64
        request_data = {"search_type": "new_game", "search_input": game_data}
    return request_data


def load_image_with_extensions(game_id, images_path, extensions):
    for ext in extensions:
        image = cv2.imread(os.path.join(images_path, f"{game_id}.{ext}"))
        if image is not None:
            return image
    return None


def visualise_game_search_registry_results(game_id, grid_size, images_path, max_similar_games, response):
    query_image = load_image_with_extensions(game_id, images_path, ["jpg", "jpeg", "png"])

    if query_image is not None:
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
        found_games = response.json()[:max_similar_games]
        print(found_games)

        _, axes = plt.subplots(grid_size + 1, grid_size, figsize=(10, 10))
        axes[0, 0].imshow(np.full_like(query_image, 255))
        axes[0, 0].axis("off")
        axes[0, 1].imshow(query_image)
        axes[0, 1].axis("off")
        axes[0, 2].imshow(np.full_like(query_image, 255))
        axes[0, 2].axis("off")

        for match, ax in zip(found_games, axes[1:].flatten()):
            image = load_image_with_extensions(match['id'], images_path, ["jpg", "jpeg", "png"])
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ax.imshow(image)
                ax.axis("off")
        plt.show()


def visualise_text_search_results(args, images_path, response):
    found_games = response.json()
    print(found_games)
    grid_size = int(math.sqrt(len(found_games)))
    found_games = found_games[:grid_size**2]
    _, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    for match, ax in zip(found_games, axes.flatten()):
        image_ext = None
        for ext in ["jpg", "jpeg", "png"]:
            image = cv2.imread(os.path.join(images_path, f"{match['id']}.{ext}"))
            if image is not None:
                image_ext = ext
                break

        if image_ext is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
            ax.axis("off")
    plt.suptitle(f"Query : {args.search_text}")
    plt.show()


def main():
    args = get_args()
    api, game_id, request_data = prepare_request(args)

    if not request_data:
        raise ValueError(f"Invalid search option: {args.search_option}")

    start = time.time()
    response = requests.post(api, data=json.dumps(request_data), headers={"Content-Type": "application/json"})
    print(f"[{args.search_option.upper()}] - Response time: {time.time() - start} seconds")
    images_path = os.path.join(args.data_path, "images")

    if args.search_option == "text":
        visualise_text_search_results(args, images_path, response)

    elif args.search_option == "image":
        print(response.json())
    else:
        max_similar_games = 9
        grid_size = int(math.sqrt(max_similar_games))
        visualise_game_search_registry_results(game_id, grid_size, images_path, max_similar_games, response)


if __name__ == '__main__':
    main()
