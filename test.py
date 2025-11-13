import argparse
import os.path
import random
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import faiss
import matplotlib.pyplot as plt
import numpy as np
import textdistance as td
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizer, PreTrainedTokenizer

from multimedia_search.rule_engine.engine import MetadataRuleEngine
from multimedia_search.semantic_search_engine.data.dataset import MultiMediaDataset
from multimedia_search.semantic_search_engine.embedding_ops import get_embedding_pairs, get_faiss_index, \
    get_image_embeddings, \
    get_pairwise_fast, get_text_embeddings
from multimedia_search.semantic_search_engine.modelling.multi_domain_model import MultiMediaCLIPModel
from multimedia_search.utility.data_ops import get_metadata


def find_matches(model: nn.Module,
                 tokenizer: PreTrainedTokenizer,
                 target_embeddings: torch.Tensor,
                 query: str,
                 image_filenames: List[str],
                 config: Dict[str, Any],
                 n: int = 9,
                 normalize: bool = False,
                 vis: bool = False) -> None:
    """

    :param model:
    :param tokenizer:
    :param target_embeddings:
    :param query:
    :param image_filenames:
    :param config:
    :param n:
    :param normalize:
    :param vis:
    :return:
    """
    images_path = os.path.join(config["data"]["dataset_path"], "images")
    encoded_query = tokenizer([query])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = {
        key: torch.tensor(values).to(device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)

    if normalize:
        target_embeddings = F.normalize(target_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings @ target_embeddings.T

    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]

    if vis:
        probs = values[::5].cpu().numpy().tolist()
        _, axes = plt.subplots(3, 3, figsize=(10, 10))
        for match, ax, prob in zip(matches, axes.flatten(), probs):
            image = cv2.imread(os.path.join(images_path, match))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
            ax.set_title(f"Sim score: {prob:.5f}")
            ax.axis("off")
        plt.suptitle(f"Query : {query}")
        plt.show()


def find_matches_faiss(model: nn.Module,
                       tokenizer: PreTrainedTokenizer,
                       faiss_index: faiss.IndexFlat,
                       query: str,
                       image_filenames: List[str],
                       config: Dict[str, Any],
                       n: int = 9,
                       vis: bool = False) -> None:
    """

    :param model:
    :param tokenizer:
    :param faiss_index:
    :param query:
    :param image_filenames:
    :param config:
    :param n:
    :param vis:
    :return:
    """
    images_path = os.path.join(config["data"]["dataset_path"], "images")
    encoded_query = tokenizer([query])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = {
        key: torch.tensor(values).to(device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features).cpu().numpy()

    text_embeddings = np.float32(text_embeddings).reshape(1, -1)
    # faiss.normalize_L2(text_embeddings)
    probs, indices = faiss_index.search(text_embeddings, n)

    matches = [image_filenames[idx] for idx in indices[0]]

    if vis:
        _, axes = plt.subplots(3, 3, figsize=(10, 10))
        for match, ax, prob in zip(matches, axes.flatten(), probs[0]):
            image = cv2.imread(os.path.join(images_path, match))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
            ax.set_title(f"Sim score: {prob:.5f}")
            ax.axis("off")
        plt.suptitle(f"Query : {query}")

        plt.show()


def find_similar(test_embeddings: List[List[torch.Tensor]],
                 image_filenames: List[str],
                 annotations: List[Dict[str, Any]],
                 config: Dict[str, Any],
                 n: int = 9,
                 index_to_search: int = 0,
                 output_folder_path: Optional[str] = None) -> None:
    """

    :param test_embeddings:
    :param image_filenames:
    :param annotations:
    :param config:
    :param n:
    :param index_to_search:
    :param output_folder_path:
    :return:
    """
    images_path = os.path.join(config["data"]["dataset_path"], "images")

    query_annotations = annotations[index_to_search]
    query_text_embeddings = test_embeddings[0][index_to_search]
    query_image_embeddings = test_embeddings[1][index_to_search]

    similarity_scores = np.array([])

    sim_records = []

    for src_text_embedding, src_image_embedding, src_annotation in tqdm(zip(test_embeddings[0], test_embeddings[1],
                                                                            annotations)):
        text_sim_score = F.cosine_similarity(query_text_embeddings, src_text_embedding).item()
        image_sim_score = F.cosine_similarity(query_image_embeddings, src_image_embedding).item()

        genre_sim_score = td.Tversky(ks=[0.8, 0.2])(query_annotations["genres"],
                                                    src_annotation["genres"])
        platform_sim_score = float(td.sorensen.normalized_similarity(query_annotations["platforms"],
                                                                     src_annotation["platforms"]) > 0.0)

        aggregate_sim_score = (text_sim_score * 1.0 + image_sim_score * 0.2
                               + genre_sim_score * 0.6 + platform_sim_score)

        sim_records.append({
            "text_sim_score": text_sim_score,
            "image_sim_score": image_sim_score,
            "genre_sim_score": genre_sim_score,
            "platform_sim_score": platform_sim_score,
            "aggregate_sim_score": aggregate_sim_score
        })

        similarity_scores = np.append(similarity_scores, aggregate_sim_score)

    # Find the indexes of the most similar n pairs
    indices = np.argsort(similarity_scores)[-n:][::-1]
    query_image = cv2.imread(os.path.join(images_path, image_filenames[index_to_search]))

    matches = [image_filenames[idx] for idx in indices]
    match_sim_records = [sim_records[idx] for idx in indices]
    match_names = [annotations[idx]["product_name"] for idx in indices]

    pprint(query_annotations)
    pprint(list(zip(match_names, match_sim_records)))

    # # Plot the query image
    plt.imshow(query_image)

    # # Plot the most similar images
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(os.path.join(images_path, match))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")
    if output_folder_path is not None:
        plt.savefig(os.path.join(output_folder_path, f"sim_search_{query_annotations['product_name']}.png"))
    else:
        plt.show()


def find_in_cached(similarity_graph: Dict[int, Dict],
                   search_indices: List[int],
                   images_folder: str, output_folder_path: str,
                   n: int = 9) -> None:
    """

    :param similarity_graph:
    :param search_indices:
    :param images_folder:
    :param output_folder_path:
    :param n:
    :return:
    """
    for search_index in tqdm(search_indices):
        query_id = search_index
        query_name = similarity_graph[query_id]["game_name"]
        query_similar_games = similarity_graph[query_id]["similar_games"]
        query_similar_games = query_similar_games[:n]
        query_similar_games_ids = [sim_game["id"] for sim_game in query_similar_games]

        query_similar_games_names = [similarity_graph[sim_game]["game_name"] for sim_game in query_similar_games_ids]

        query_similar_games = [sim_game for sim_game in query_similar_games_names if sim_game != query_name]

        print(f"Query: {query_name}")
        print(f"Similar games: {query_similar_games}")
        print("")

        query_image = cv2.imread(os.path.join(images_folder, f"{query_id}.jpg"))

        # # Plot the most similar images
        _, axes = plt.subplots(4, 3, figsize=(10, 10))
        # Add query image
        axes[0, 0].imshow(np.full_like(query_image, 255))
        axes[0, 0].axis("off")
        axes[0, 1].imshow(query_image)
        axes[0, 1].axis("off")
        axes[0, 2].imshow(np.full_like(query_image, 255))
        axes[0, 2].axis("off")

        for match, ax in zip(query_similar_games_ids, axes[1:].flatten()):
            image = cv2.imread(os.path.join(images_folder, f"{match}.jpg"))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
            ax.axis("off")

        if output_folder_path is not None:
            plt.savefig(os.path.join(output_folder_path, f"sim_search_{query_name}.png"))
        else:
            plt.show()


def prepare_embeddings(model: nn.Module,
                       test_data_loader: DataLoader,
                       search_in: str,
                       device: str) -> Union[torch.Tensor, List[List[torch.Tensor]]]:
    """

    :param model:
    :param test_data_loader:
    :param search_in:
    :param device:
    :return:
    """
    if search_in == "image":
        if os.path.exists("cached_data/test_embeddings_image.pt"):
            test_embeddings = torch.load("cached_data/test_embeddings_image.pt")
        else:
            test_embeddings = get_image_embeddings(model, test_data_loader, device=device)
            torch.save(test_embeddings, "cached_data/test_embeddings_image.pt")
    elif search_in == "text":
        if not os.path.exists("cached_data/test_embeddings_text.pt"):
            test_embeddings = get_text_embeddings(model, test_data_loader, device=device)
            torch.save(test_embeddings, "cached_data/test_embeddings_text.pt")
        else:
            test_embeddings = torch.load("cached_data/test_embeddings_text.pt")
    elif not os.path.exists("cached_data/test_embeddings_mixed.pt"):
        test_embeddings = get_embedding_pairs(model, test_data_loader, device=device)
        torch.save(test_embeddings, "cached_data/test_embeddings_mixed.pt")
    else:
        test_embeddings = torch.load("cached_data/test_embeddings_mixed.pt")
    return test_embeddings


def prepare_model(config: Dict[str, Any],
                  model_path: str) -> Tuple[str, str, nn.Module, DataLoader, MultiMediaDataset, PreTrainedTokenizer]:
    """

    :param config:
    :param model_path:
    :return:
    """
    hyperparams = config["hyper-parameters"]
    modelling_params = config["model"]
    data_path = config["data"]["dataset_path"]

    image_transforms = A.Compose([
        A.PadIfNeeded(min_height=modelling_params["image_size"], min_width=modelling_params["image_size"], p=1.0),
        A.Resize(modelling_params["image_size"], modelling_params["image_size"], p=1.0),
        A.Normalize(p=1.0)])

    tokenizer = DistilBertTokenizer.from_pretrained(modelling_params["text_tokenizer"])

    test_dataset = MultiMediaDataset(images_folder=os.path.join(data_path, "images"),
                                     annotations_folder=os.path.join(data_path, "annotations"),
                                     img_preprocessor=image_transforms,
                                     text_tokenizer=tokenizer,
                                     max_length=modelling_params["max_length"],
                                     use_images=modelling_params["use_image_encoder"],
                                     use_videos=modelling_params["use_video_encoder"])
    test_data_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=hyperparams["num_workers"]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MultiMediaCLIPModel(model_config=modelling_params, hyperparams=hyperparams).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return data_path, device, model, test_data_loader, test_dataset, tokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", "--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("-mp", "--model_path", type=str, required=False, default="best.pt")
    parser.add_argument("-so", "--search_option", choices=["text", "image", "lazy_similarity", "pairwise"],
                        type=str, required=False, default="similarity")
    parser.add_argument("-st", "--search_text", type=str, required=False,
                        default="I want to play a story game and the story has to change based on my decisions.")
    parser.add_argument("--use_faiss", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = get_args()
    config_path = args.config
    model_path = args.model_path
    search_in = args.search_option
    use_faiss = args.use_faiss

    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    print("Preparing the data and the model!")
    data_path, device, model, test_data_loader, test_dataset, tokenizer = prepare_model(config=config,
                                                                                        model_path=model_path)
    print(f"Retrieving {search_in} embeddings!")
    test_embeddings = prepare_embeddings(model=model, test_data_loader=test_data_loader, search_in=search_in,
                                         device=device)
    print("Preprocessing is ready!")
    if search_in == "lazy_similarity":
        os.makedirs("similarity_search_results", exist_ok=True)
        annotations = get_metadata(folder_path=data_path)

        search_indices = random.sample(range(len(annotations)), 20)

        for search_index in tqdm(search_indices):
            find_similar(test_embeddings=test_embeddings,
                         image_filenames=test_dataset.image_paths,
                         annotations=annotations,
                         config=config,
                         n=9,
                         index_to_search=search_index,
                         output_folder_path="similarity_search_results")
    elif search_in == "pairwise":
        print("Preparing the metadata!")
        if not os.path.exists("cached_data/game_data.pt"):
            annotations = get_metadata(data_path)
            torch.save(annotations, "cached_data/game_data.pt")
        else:
            annotations = torch.load("cached_data/game_data.pt")
        rule_engine = MetadataRuleEngine(annotations=annotations)
        print("Preparing the similarity graph!")
        if not os.path.exists("cached_data/similarity_graph.pt") or not os.path.exists(
                "cached_data/pairwise_similarities.pt"):
            similarity_graph = get_pairwise_fast(test_embeddings=test_embeddings,
                                                 annotations=annotations,
                                                 rule_engine=rule_engine,
                                                 similarity_parameters=config["similarity_parameters"])
            # save the similarity graph and pairwise similarities
            torch.save(similarity_graph, "cached_data/similarity_graph.pt")
            # torch.save(pairwise_similarities, "cached_data/pairwise_similarities.pt")
        else:
            similarity_graph = torch.load("cached_data/similarity_graph.pt")
            pairwise_similarities = torch.load("cached_data/pairwise_similarities.pt")

        search_indices = random.sample(range(len(annotations)), 50)
        os.makedirs("pairwise_search_results", exist_ok=True)
        find_in_cached(similarity_graph=similarity_graph,
                       search_indices=search_indices,
                       images_folder=os.path.join(config["data"]["dataset_path"], "images"),
                       output_folder_path="pairwise_search_results")
    else:
        query = args.search_text

        if use_faiss:
            faiss_index_path = f"cached_data/faiss_vector_{args.search_option}.index"
            index = get_faiss_index(faiss_index_path=faiss_index_path, test_embeddings=test_embeddings)
            find_matches_faiss(
                model=model,
                tokenizer=tokenizer,
                faiss_index=index,
                query=query,
                image_filenames=test_dataset.image_paths,
                config=config,
                n=9,
                vis=True
            )
        else:
            find_matches(
                model=model,
                tokenizer=tokenizer,
                target_embeddings=test_embeddings,
                query=query,
                image_filenames=test_dataset.image_paths,
                config=config,
                n=9,
                vis=True
            )


if __name__ == '__main__':
    main()
