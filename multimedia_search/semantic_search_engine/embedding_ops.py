import copy
import os
import queue
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from multimedia_search.rule_engine.engine import MetadataRuleEngine


def get_image_embeddings(model: nn.Module, test_data_loader: DataLoader, device: str = "cuda") -> torch.Tensor:
    """

    :param model:
    :param test_data_loader:
    :param device:
    :return:
    """
    test_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_data_loader):
            image_features = model.image_encoder(batch["image"].to(device))
            image_embeddings = model.image_projection(image_features)
            test_image_embeddings.append(image_embeddings)
    return torch.cat(test_image_embeddings)


def get_text_embeddings(model: nn.Module, test_data_loader: DataLoader, device: str = "cuda") -> torch.Tensor:
    """

    :param model:
    :param test_data_loader:
    :param device:
    :return:
    """
    test_text_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_data_loader):
            text_features = model.text_encoder(
                input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)
            )
            text_embeddings = model.text_projection(text_features)
            test_text_embeddings.append(text_embeddings)
    return torch.cat(test_text_embeddings)


def get_embedding_pairs(model: nn.Module,
                        test_data_loader: DataLoader,
                        device: str = "cuda") -> List[List[torch.Tensor]]:
    """

    :param model:
    :param test_data_loader:
    :param device:
    :return:
    """
    test_text_embeddings = []
    test_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_data_loader):
            text_features = model.text_encoder(
                input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)
            )
            text_embeddings = model.text_projection(text_features)
            test_text_embeddings.append(text_embeddings)
            image_features = model.image_encoder(batch["image"].to(device))
            image_embeddings = model.image_projection(image_features)
            test_image_embeddings.append(image_embeddings)
    return [test_text_embeddings, test_image_embeddings]


def calculate_similarities(annotation_id: int, annotations: List[Dict[str, Any]], rule_engine: MetadataRuleEngine,
                           pairwise_image: np.ndarray, pairwise_text: np.ndarray, similarity_graph: Dict[int, Any],
                           similarity_parameters: Dict[str, float],
                           min_connection_score: float = 2.4, max_similar_game_size: int = 15) -> None:
    """

    :param annotation_id:
    :param annotations:
    :param rule_engine:
    :param pairwise_image:
    :param pairwise_text:
    :param similarity_graph:
    :param similarity_parameters:
    :param min_connection_score:
    :param max_similar_game_size:
    :return:
    """
    query_annotations = annotations[annotation_id]
    query_id = query_annotations["product_id"]
    if query_id not in similarity_graph:
        similarity_graph[query_id] = {
            "game_name": query_annotations["product_name"],
            "similar_games": queue.PriorityQueue(maxsize=max_similar_game_size)
        }
    for j in range(annotation_id + 1, len(annotations)):
        src_annotation = annotations[j]
        src_id = src_annotation["product_id"]

        text_sim_score = pairwise_text[annotation_id][j]
        image_sim_score = pairwise_image[annotation_id][j]

        genre_sim_score, platform_sim_score, mode_sim_score, theme_sim_score = rule_engine.calc_similarity(
            query_annotations, src_annotation)

        aggregate_sim_score = (text_sim_score * similarity_parameters["text_similarity_weight"] +
                               image_sim_score * similarity_parameters["image_similarity_weight"] +
                               + genre_sim_score * similarity_parameters["genre_similarity_weight"] +
                               + platform_sim_score * similarity_parameters["platform_similarity_weight"])
        if mode_sim_score >= 0.0:
            aggregate_sim_score += mode_sim_score * similarity_parameters["mode_similarity_weight"]
        if theme_sim_score >= 0.0:
            aggregate_sim_score += theme_sim_score * similarity_parameters["theme_similarity_weight"]

        if aggregate_sim_score > min_connection_score:
            src_similarity_data = {
                "id": src_id,
                "similarity_score": aggregate_sim_score
            }

            if similarity_graph[query_id]["similar_games"].full():
                similarity_graph[query_id]["similar_games"].get()  # remove the lowest score
            similarity_graph[query_id]["similar_games"].put(
                (aggregate_sim_score, id(src_similarity_data), src_similarity_data))

            if src_id not in similarity_graph:
                similarity_graph[src_id] = {
                    "game_name": src_annotation["product_name"],
                    "similar_games": queue.PriorityQueue(maxsize=max_similar_game_size)
                }

            query_similarity_data = copy.deepcopy(src_similarity_data)
            query_similarity_data["id"] = query_id

            if similarity_graph[src_id]["similar_games"].full():
                similarity_graph[src_id]["similar_games"].get()  # remove the lowest score
            similarity_graph[src_id]["similar_games"].put(
                (aggregate_sim_score, id(query_similarity_data), query_similarity_data))


def get_pairwise_fast(test_embeddings: List[List[torch.Tensor]],
                      annotations: List[Dict[str, Any]],
                      rule_engine: MetadataRuleEngine,
                      similarity_parameters: Dict[str, Union[float, int]],
                      folder_to_save: Optional[str] = "cached_data_new",
                      caching_rate: Optional[int] = 30000,
                      debug: bool = False) -> Dict[int, Any]:
    """

    :param test_embeddings:
    :param annotations:
    :param rule_engine:
    :param similarity_parameters:
    :param folder_to_save:
    :param caching_rate:
    :param debug:
    :return:
    """
    print("Calculating pairwise similarities for text embeddings")
    test_embeddings_text = torch.cat(test_embeddings[0]).cpu().numpy()
    print(f"The shape of the test text embeddings is {test_embeddings_text.shape}")
    pairwise_text = cosine_similarity(test_embeddings_text, test_embeddings_text)
    print(f"The shape of the pairwise text similarities is {pairwise_text.shape}")

    print("Calculating pairwise similarities for image embeddings")
    test_embeddings_image = torch.cat(test_embeddings[1]).cpu().numpy()
    print(f"The shape of the test image embeddings is {test_embeddings_image.shape}")
    pairwise_image = cosine_similarity(test_embeddings_image, test_embeddings_image)
    print(f"The shape of the pairwise image similarities is {pairwise_image.shape}")

    cached_sim_graph_path = os.path.join(folder_to_save, "sim_graph_fast")
    if not os.path.exists(cached_sim_graph_path) or len(os.listdir(cached_sim_graph_path)) == 0:
        os.makedirs(cached_sim_graph_path, exist_ok=True)
        iteration_start = 0
        similarity_graph = {}

    else:
        saved_sim_file = os.listdir(cached_sim_graph_path)[0]
        previous_iteration = os.path.basename(saved_sim_file).split(".pt")[0].split("_")[-1]
        similarity_graph = torch.load(os.path.join(cached_sim_graph_path, saved_sim_file))
        iteration_start = int(previous_iteration) + 1

    print(f"Starting from iteration {iteration_start}")
    for annotation_id in tqdm(range(iteration_start, len(annotations))):
        calculate_similarities(annotation_id=annotation_id,
                               annotations=annotations,
                               rule_engine=rule_engine,
                               pairwise_image=pairwise_image,
                               pairwise_text=pairwise_text,
                               similarity_graph=similarity_graph,
                               similarity_parameters=similarity_parameters,
                               min_connection_score=similarity_parameters["aggregate_similarity_threshold"],
                               max_similar_game_size=similarity_parameters["max_similar_game_size"])

        if debug and (annotation_id % 1000 == 0 and annotation_id > 0):
            random_game_ids = np.random.choice(list(similarity_graph.keys()), size=5, replace=False)
            for game_id in random_game_ids:
                print(f"Game name : {similarity_graph[game_id]['game_name']}")
                p_id_game_list = list(similarity_graph[game_id]["similar_games"].queue)
                similar_games = [p_id_elem[2] for p_id_elem in p_id_game_list]
                sorted_similar_games = sorted(similar_games, key=lambda x: x["similarity_score"], reverse=True)
                print("|" * 50)
                for similar_game in sorted_similar_games:
                    similar_game_id = similar_game['id']
                    print(f"Similar game name : {similarity_graph[similar_game_id]['game_name']} - "
                          f"{similar_game['similarity_score']}")
                    print("----------------------------------------------\n")
                print("|" * 50)

        # Cache the similarity graph every N iterations
        if annotation_id % caching_rate == 0 and annotation_id > 0:
            # write query similarity data to json
            previous_iteration_path = os.path.join(cached_sim_graph_path,
                                                   f'sim_graph_iteration_{annotation_id - caching_rate}.json')
            os.system(f"rm -rf {previous_iteration_path}")
            torch.save(similarity_graph, os.path.join(cached_sim_graph_path, f"sim_graph_iteration_{annotation_id}.pt"))

    for product_id, similar_games_info in similarity_graph.items():
        # Get the games in the reverse order - Highest similarity score first
        p_id_game_list = list(similar_games_info["similar_games"].queue)
        similar_games = [p_id_elem[2] for p_id_elem in p_id_game_list]
        sorted_similar_games = sorted(similar_games, key=lambda x: x["similarity_score"], reverse=True)
        similarity_graph[product_id]["similar_games"] = sorted_similar_games

    return similarity_graph


def get_faiss_index(faiss_index_path: str,
                    test_embeddings: torch.Tensor,
                    comparison: str = "cosine") -> faiss.IndexFlat:
    """

    :param faiss_index_path:
    :param test_embeddings:
    :param comparison:
    :return:
    """
    if not os.path.exists(faiss_index_path):
        if comparison == "cosine":
            index = faiss.IndexFlatIP(256)
        else:
            index = faiss.IndexFlatL2(256)
        for embedding in tqdm(test_embeddings):
            vector = embedding.detach().cpu().numpy().reshape(1, -1)
            vector = np.float32(vector)
            # faiss.normalize_L2(vector)
            index.add(vector)

        faiss.write_index(index, faiss_index_path)
    else:
        index = faiss.read_index(faiss_index_path)
    return index


class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.labels = None
        self.k = k

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype(np.float32))
        self.labels = labels

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        distances, indices = self.index.search(embeddings.astype(np.float32), k=self.k)
        votes = self.labels[indices]
        return np.array([np.argmax(np.bincount(x)) for x in votes])

    def predict_most_similar(self, embeddings: np.ndarray) -> List[Tuple[str, int]]:
        distances, indices = self.index.search(embeddings.astype(np.float32), k=self.k * self.k)
        votes = self.labels[indices]
        occurances = Counter(votes[0].tolist())
        return occurances.most_common(self.k)

    def save(self, output_path: str) -> None:
        faiss.write_index(self.index, output_path)
        folder_name = os.path.dirname(output_path)
        torch.save(self.labels, os.path.join(folder_name, "sentence_labels.pt"))

    def load(self, faiss_path: str) -> None:
        self.index = faiss.read_index(faiss_path)
        self.labels = torch.load(os.path.join(os.path.dirname(faiss_path), "sentence_labels.pt"))
