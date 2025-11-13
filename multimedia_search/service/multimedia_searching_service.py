import base64
import copy
import logging
import os
import queue
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import faiss
import numpy as np
import torch
import yaml
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import DistilBertTokenizer
from ts.torch_handler.base_handler import BaseHandler

from multimedia_search.rule_engine.engine import MetadataRuleEngine
from multimedia_search.semantic_search_engine.embedding_ops import get_faiss_index
from multimedia_search.semantic_search_engine.modelling.multi_domain_model import MultiMediaCLIPModel
from multimedia_search.utility.data_ops import get_metadata


class MultiMediaSearchHandler(BaseHandler, ABC):
    """
    MultiMediaSearchHandler class is a custom handler for TorchServe.
    :param logger: Logger for the handler
    :param overwrite_cache: Whether to overwrite cached data or not
    """

    def __init__(self, logger: logging.Logger, overwrite_cache: bool = False) -> None:
        super(MultiMediaSearchHandler, self).__init__()
        self.logger = logger
        self.overwrite_cache: bool = overwrite_cache

        self.initialized = False
        self.n_similar: int = 9
        self.min_connection_score: float = 2.4

        self.config: Optional[Dict[str, Any]] = None
        self.hyperparams: Optional[Dict[str, Any]] = None
        self.modelling_params: Optional[Dict[str, Any]] = None
        self.similarity_parameters: Optional[Dict[str, Any]] = None
        self.data_path: Optional[str] = None
        self.image_processor: Optional[A.Compose] = None
        self.tokenizer: Optional[DistilBertTokenizer] = None
        self.device: Optional[torch.device] = None
        self.model: Optional[MultiMediaCLIPModel] = None
        self.rule_engine: Optional[MetadataRuleEngine] = None
        self.embeddings: Optional[List[List[torch.Tensor]]] = []
        self.game_data: Optional[List[Dict[str, Any]]] = None
        self.similarity_graph: Optional[Dict[int, Any]] = None
        self.faiss_index_text: Optional[faiss.IndexFlat] = None
        self.faiss_index_image: Optional[faiss.IndexFlat] = None

        self.config_path: Optional[str] = None
        self.model_path: Optional[str] = None
        self.cached_data_folder: Optional[str] = None

    def initialize(self, ctx: Any, *args, **kwargs) -> None:
        """

        :param ctx:
        :param args:
        :param kwargs:
        :return:
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        self.logger.info(os.listdir(model_dir))

        self.config_path = os.path.join(model_dir, "config_v1.yaml")
        self.model_path = os.path.join(model_dir, "best.pt")
        self.cached_data_folder = model_dir

        self.config = yaml.load(open(self.config_path), Loader=yaml.FullLoader)
        self.hyperparams = self.config["hyper-parameters"]
        self.modelling_params = self.config["model"]
        self.data_path = self.config["data"]["dataset_path"]
        self.similarity_parameters = self.config["similarity_parameters"]

        self.n_similar = self.similarity_parameters["max_similar_game_size"]

        self._initialize_from_cached()

        self._initialize_model(self.model_path)

        self.game_data = get_metadata(self.data_path)
        self.logger.info("Loaded metadata successfully")
        self.rule_engine = MetadataRuleEngine(self.game_data)
        self.logger.info("Initialized rule engine successfully")

        self.logger.info(f"Initialized {self.__class__.__name__} successfully")

        self.initialized = True

    def reinitialize(self) -> None:
        """
        Reinitializes the model on reload
        :return:
        """
        self._initialize_from_cached()
        self.game_data = get_metadata(self.data_path)
        self.rule_engine = MetadataRuleEngine(self.game_data)

    def _initialize_model(self, model_path):
        self.image_processor = A.Compose([
            A.PadIfNeeded(min_height=self.modelling_params["image_size"],
                          min_width=self.modelling_params["image_size"],
                          p=1.0),
            A.Resize(self.modelling_params["image_size"], self.modelling_params["image_size"], p=1.0),
            A.Normalize(p=1.0)])
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.modelling_params["text_tokenizer"])
        self.logger.info("Loaded tokenizer and image processor successfully")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Read model serialize/pt file
        self.model = MultiMediaCLIPModel(model_config=self.modelling_params, hyperparams=self.hyperparams).to(
            self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.logger.info("Loaded model successfully")

    def _initialize_from_cached(self) -> None:
        self.similarity_graph = torch.load(os.path.join(self.cached_data_folder, "similarity_graph.pt"))
        self.logger.info("Loaded similarity graph successfully")
        self.embeddings = torch.load(os.path.join(self.cached_data_folder, "test_embeddings_mixed.pt"))
        self.logger.info("Loaded embeddings successfully")
        faiss_index_path = os.path.join(self.cached_data_folder, "faiss_vector_text.index")

        test_text_embeddings = torch.load(os.path.join(self.cached_data_folder, "test_embeddings_text.pt"))
        self.faiss_index_text = get_faiss_index(faiss_index_path=faiss_index_path, test_embeddings=test_text_embeddings)
        self.logger.info("Loaded cached faiss index text data successfully")

        faiss_index_path = os.path.join(self.cached_data_folder, "faiss_vector_image.index")
        test_image_embeddings = torch.load(os.path.join(self.cached_data_folder, "test_embeddings_image.pt"))

        self.faiss_index_image = get_faiss_index(faiss_index_path=faiss_index_path,
                                                 test_embeddings=test_image_embeddings)
        self.logger.info("Loaded cached faiss index image data successfully")
        self.logger.info("Loaded cached data successfully")

    def preprocess(self, data: Any) -> Any:
        """

        :param data:
        :return:
        """
        for row in data:
            query = row.get("data") or row.get("body")
            search_type = query.get("search_type")
            search_input = query.get("search_input")

            if search_type == "text":
                search_input = str(search_input)
                self.logger.info(f"[MULTIMODEL] Received Query : {query}")
                return self.preprocess_text_data(search_input)
            elif search_type == "image":
                self.logger.info(f"[MULTIMODEL] Received Query : {query}")
                return self.preprocess_image_data(search_input)
            elif search_type == "game":
                self.logger.info(f"[MULTIMODEL] Received Query : {query}")
                return self.preprocess_game_data(search_input)
            elif search_type == "new_game":
                image = query["search_input"].pop("image", None)
                self.logger.info(f"[MULTIMODEL] Received Query : {query}")
                if image is not None:
                    query["search_input"]["image"] = image
                return self.preprocess_full_game_data(search_input)
            else:
                raise ValueError(f"Invalid search type: {search_type}")

    def inference(self, preprocessed_inputs: Union[Dict[str, Any], int], **kwargs
                  ) -> Union[List[Dict[str, Any]], List[int]]:
        """

        :param preprocessed_inputs:
        :param kwargs:
        :return:
        """
        if isinstance(preprocessed_inputs, int):
            return self._find_similar_games(preprocessed_inputs)
        elif isinstance(preprocessed_inputs, dict):
            if "input_ids" in preprocessed_inputs:
                self.logger.info("Inferring with text")
                return self._search_game_with_text(preprocessed_inputs)
            elif "image" in preprocessed_inputs:
                self.logger.info("Inferring with image")
                return self._search_game_with_image(preprocessed_inputs)
            elif "game_id" in preprocessed_inputs:
                self.logger.info(f"Received game id : {preprocessed_inputs['game_id']}")
                return self._find_similar_games(preprocessed_inputs["game_id"])
            elif "new_game" in preprocessed_inputs:
                self.logger.info(f"Registering a new game named {preprocessed_inputs['new_game']['product_name']}")
                game_id = self._add_new_game(preprocessed_inputs["new_game"])
                similar_games = self._find_similar_games(game_id)
                self.logger.info(f"Found similar games: {similar_games}")
                return similar_games
            else:
                raise ValueError(f"Invalid input: {preprocessed_inputs}")

    def postprocess(self, inference_output: Union[List[Dict[str, Any]], List[int]], **kwargs
                    ) -> Union[List[List[Dict[str, Any]]], List[List[int]]]:
        """

        :param inference_output:
        :param kwargs:
        :return:
        """
        return [inference_output]

    def preprocess_text_data(self, text: str) -> Dict[str, torch.Tensor]:
        """

        :param text:
        :return:
        """
        encoded_text = self.tokenizer([text])
        return {
            key: torch.tensor(values).to(self.device)
            for key, values in encoded_text.items()
        }

    def preprocess_image_data(self, image: str) -> Dict[str, torch.Tensor]:
        image = base64.b64decode(image)
        if isinstance(image, (bytearray, bytes)):
            image = np.frombuffer(image, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_processor(image=image)["image"]
        image = torch.tensor(image).permute(2, 0, 1).float()

        return {
            "image": image.to(self.device)
        }

    def preprocess_game_data(self, game_id: int) -> Dict[str, int]:
        return {
            "game_id": game_id
        }

    def preprocess_full_game_data(self, game_id: Dict[str, Any]) -> Dict[str, Any]:
        image = game_id.get("image")
        if image is None:
            raise ValueError("Image is not provided")
        preprocessed_image = self.preprocess_image_data(image)["image"]
        summary = game_id.get("summary")
        if summary is None:
            raise ValueError("Summary is not provided")
        preprocessed_text = self.preprocess_text_data(summary)

        return {
            "new_game": {
                "preprocessed_image": preprocessed_image,
                "preprocessed_text": preprocessed_text,
                "summary": summary,
                "product_id": game_id.get("product_id"),
                "product_name": game_id.get("product_name"),
                "genres": game_id.get("genres"),
                "platforms": game_id.get("platforms"),
                "themes": game_id.get("themes", []),
                "modes": game_id.get("modes", [])
            }
        }

    def _search_game_with_text(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        with torch.no_grad():
            text_features = self.model.text_encoder(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            text_embeddings = self.model.text_projection(text_features).cpu().numpy()

        text_embeddings = np.float32(text_embeddings).reshape(1, -1)
        probs, indices = self.faiss_index_text.search(text_embeddings, self.n_similar)

        return [
            {
                "id": self.game_data[idx]["product_id"],
                "name": self.game_data[idx]["product_name"],
            }
            for idx in indices[0]
        ]

    def _search_game_with_image(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        with torch.no_grad():
            image_features = self.model.image_encoder(inputs["image"])
            image_embeddings = self.model.image_projection(image_features).cpu().numpy()

        image_embeddings = np.float32(image_embeddings).reshape(1, -1)
        # faiss.normalize_L2(image_embeddings)
        probs, indices = self.faiss_index_image.search(image_embeddings, self.n_similar)

        return [
            {
                "id": self.game_data[idx]["product_id"],
                "name": self.game_data[idx]["product_name"],
            }
            for idx in indices[0]
        ]

    def _find_similar_games(self, game_id: int) -> List[int]:
        """
        :param game_id:
        :return:
        """
        return self.similarity_graph[game_id]["similar_games"][:self.n_similar]

    def _add_new_game(self, game_info: Dict[str, Any]) -> int:
        query_image_embeddings, query_text_embeddings = self.get_game_embeddings(game_info)

        query_annotations = {
            "genres": game_info["genres"],
            "platforms": game_info["platforms"],
            "themes": game_info.get("themes", []),
            "modes": game_info.get("modes", [])
        }

        query_id = game_info["product_id"]

        if query_id in self.similarity_graph:
            self.logger.info(f"Game with id {query_id} already exists")
            return query_id

        self.similarity_graph[query_id] = {
            "game_name": game_info["product_name"],
            "similar_games": queue.PriorityQueue(maxsize=self.n_similar)
        }

        self.logger.info(f"Text Embeddings Length: {len(self.embeddings[0])}\n"
                         f"Image Embeddings Length: {len(self.embeddings[1])}\n"
                         f"Game Data Length: {len(self.game_data)}\n")

        image_sim_scores, text_sim_scores = self.get_similarity_scores_per_query(query_image_embeddings,
                                                                                 query_text_embeddings)

        self.logger.info("Updating similarity graph...")

        # Update similarity graph
        for annotation_idx, src_annotation in tqdm(enumerate(self.game_data), total=len(self.game_data)):
            aggregate_sim_score = self.calculate_similarity_score(query_annotations, src_annotation, image_sim_scores,
                                                                  text_sim_scores, annotation_idx)

            self.register_similar_games(src_annotation, query_id, aggregate_sim_score)

        self.update_data_structures(game_info, query_id, query_image_embeddings, query_text_embeddings)

        # Save data
        if self.overwrite_cache:
            torch.save(self.similarity_graph, os.path.join(self.cached_data_folder, "similarity_graph.pt"))
            torch.save(self.embeddings, os.path.join(self.cached_data_folder, "test_embeddings_mixed.pt"))
            faiss.write_index(self.faiss_index_text,
                              os.path.join(self.cached_data_folder, "faiss_vector_text.index"))
            faiss.write_index(self.faiss_index_image,
                              os.path.join(self.cached_data_folder, "faiss_vector_image.index"))
            torch.save(self.game_data, os.path.join(self.cached_data_folder, "game_data.pt"))

        return query_id

    def get_similarity_scores_per_query(self, query_image_embeddings: torch.Tensor,
                                        query_text_embeddings: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """

        :param query_image_embeddings:
        :param query_text_embeddings:
        :return:
        """
        text_embeddings = torch.cat(self.embeddings[0], dim=0).cpu().numpy()
        image_embeddings = torch.cat(self.embeddings[1], dim=0).cpu().numpy()

        query_text_embeddings_np = query_text_embeddings.cpu().numpy()
        query_image_embeddings_np = query_image_embeddings.cpu().numpy()

        # cosine similarity for image and text
        text_sim_scores = cosine_similarity(text_embeddings, query_text_embeddings_np)
        self.logger.info(f"Text similarity scores shape: {text_sim_scores.shape}")

        image_sim_scores = cosine_similarity(image_embeddings, query_image_embeddings_np)
        self.logger.info(f"Image similarity scores shape: {image_sim_scores.shape}")

        return image_sim_scores, text_sim_scores

    def get_game_embeddings(self, game_info: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param game_info:
        :return:
        """
        with torch.no_grad():
            preprocessed_text = game_info["preprocessed_text"]
            text_features = self.model.text_encoder(
                input_ids=preprocessed_text["input_ids"],
                attention_mask=preprocessed_text["attention_mask"]
            )
            query_text_embeddings: torch.Tensor = self.model.text_projection(text_features)

            preprocessed_image = game_info["preprocessed_image"]
            image_features: torch.Tensor = self.model.image_encoder(preprocessed_image.unsqueeze(0))
            query_image_embeddings = self.model.image_projection(image_features)
        return query_image_embeddings, query_text_embeddings

    def update_data_structures(self,
                               game_info: Dict[str, Any],
                               query_id: int,
                               query_image_embeddings: torch.Tensor,
                               query_text_embeddings: torch.Tensor) -> None:
        """

        :param game_info:
        :param query_id:
        :param query_image_embeddings:
        :param query_text_embeddings:
        :return:
        """
        # Update similarity graph for query
        p_id_game_list = list(self.similarity_graph[query_id]["similar_games"].queue)
        similar_games = [p_id_elem[2] for p_id_elem in p_id_game_list]
        sorted_similar_games = list(sorted(similar_games, key=lambda x: x["similarity_score"], reverse=True))
        self.similarity_graph[query_id]["similar_games"] = sorted_similar_games
        self.logger.info("Similarity graph is updated successfully")

        # Update embeddings
        self.logger.info("Updating embeddings...")
        self.embeddings[0].append(query_text_embeddings)
        self.embeddings[1].append(query_image_embeddings)

        # Update faiss index
        self.logger.info("Updating faiss indexes...")
        self.faiss_index_text.add(np.float32(query_text_embeddings.cpu().numpy()).reshape(1, -1))
        self.faiss_index_image.add(np.float32(query_image_embeddings.cpu().numpy()).reshape(1, -1))

        # Update game data
        self.logger.info("Updating game data...")
        self.game_data.append({
            "product_id": query_id,
            "product_name": game_info["product_name"],
            "summary": game_info["summary"],
            "genres": game_info["genres"],
            "platforms": game_info["platforms"]
        })

    def register_similar_games(self, src_annotation: Dict[str, Any], query_id: int, aggregate_sim_score: float) -> None:
        """

        :param src_annotation:
        :param query_id:
        :param aggregate_sim_score:
        :return:
        """
        src_id = src_annotation["product_id"]
        if aggregate_sim_score > self.similarity_parameters["aggregate_similarity_threshold"]:
            similarity_data_query = {
                "id": src_id,
                "similarity_score": aggregate_sim_score
            }

            # Update similarity graph for query
            if self.similarity_graph[query_id]["similar_games"].full():
                self.similarity_graph[query_id]["similar_games"].get()
            self.similarity_graph[query_id]["similar_games"].put(
                (aggregate_sim_score, id(similarity_data_query), similarity_data_query))

            # Update similarity graph for source
            src_insertion_index = self._find_insertion_index(src_id, aggregate_sim_score)
            similarity_data_src = copy.deepcopy(similarity_data_query)
            similarity_data_src["id"] = query_id
            self.similarity_graph[src_id]["similar_games"].insert(src_insertion_index, similarity_data_src)
            clipped_similar_games = self.similarity_graph[src_id]["similar_games"][:self.n_similar]
            self.similarity_graph[src_id]["similar_games"] = clipped_similar_games

    def calculate_similarity_score(self,
                                   query_annotations: Dict[str, Any],
                                   src_annotation: Dict[str, Any],
                                   image_sim_scores: np.ndarray,
                                   text_sim_scores: np.ndarray,
                                   annotation_idx: int) -> float:
        """

        :param query_annotations:
        :param src_annotation:
        :param image_sim_scores:
        :param text_sim_scores:
        :param annotation_idx:
        :return:
        """
        text_sim_score = text_sim_scores[annotation_idx][0]
        image_sim_score = image_sim_scores[annotation_idx][0]
        genre_sim_score, platform_sim_score, mode_sim_score, theme_sim_score = self.rule_engine.calc_similarity(
            query_annotations, src_annotation)
        aggregate_sim_score = (text_sim_score * self.similarity_parameters["text_similarity_weight"] +
                               image_sim_score * self.similarity_parameters["image_similarity_weight"] +
                               genre_sim_score * self.similarity_parameters["genre_similarity_weight"] +
                               platform_sim_score * self.similarity_parameters["platform_similarity_weight"])
        if mode_sim_score >= 0.0:
            aggregate_sim_score += mode_sim_score * self.similarity_parameters["mode_similarity_weight"]
        if theme_sim_score >= 0.0:
            aggregate_sim_score += theme_sim_score * self.similarity_parameters["theme_similarity_weight"]

        return aggregate_sim_score

    def _find_insertion_index(self, src_id: int, aggregate_sim_score: float) -> int:
        """
        Find the index of the first game that has a lower similarity score than the new game
        and insert the new game before that game. Use binary search to speed up the process
        since the list is already sorted
        :param src_id:
        :param aggregate_sim_score:
        :return:
        """
        left = 0
        right = len(self.similarity_graph[src_id]["similar_games"]) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.similarity_graph[src_id]["similar_games"][mid]["similarity_score"] < aggregate_sim_score:
                right = mid - 1
            else:
                left = mid + 1
        return left
