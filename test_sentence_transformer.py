import argparse
import os

import cv2
import torch
from matplotlib import pyplot as plt

from multimedia_search.semantic_search_engine.modelling.sentence_model import (MultiMediaSentenceModel,
                                                                               get_data_from_annotations)


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-q", "--query", type=str, default="Lego game with marvel characters")
    return arg_parser.parse_args()


def main():
    parser = parse_args()
    images_path = "gamegator_dataset_v1/images"
    annotations = torch.load("cached_data/game_data.pt")
    model = MultiMediaSentenceModel(model_config={'model_name_or_path': 'all-MiniLM-L6-v2'})

    model.initialize(annotations, cached_faiss_path="cached_data/sentence_model_faiss.index")

    sentence = parser.query

    similar_games = model.predict_most_similar(sentence)

    matches = [f"{similar_game[0]}.jpg" for similar_game in similar_games]

    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(os.path.join(images_path, match))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")
    plt.suptitle(f"Query : {sentence}")

    plt.show()


if __name__ == '__main__':
    main()
