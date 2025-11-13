import logging
import os
from abc import ABC
from typing import Any, Dict, List, Optional, Union

import torch
import yaml
from ts.torch_handler.base_handler import BaseHandler

from multimedia_search.semantic_search_engine.modelling.sentence_model import MultiMediaSentenceModel
from multimedia_search.utility.data_ops import get_metadata


class SentenceBasedSearchHandler(BaseHandler, ABC):
    """
    MultiMediaSearchHandler class is a custom handler for TorchServe.
    :param logger: Logger for the handler
    """

    def __init__(self, logger: logging.Logger) -> None:
        super(SentenceBasedSearchHandler, self).__init__()
        self.initialized = False
        self.n_similar: int = 9
        self.logger = logger

        self.config: Optional[Dict[str, Any]] = None
        self.data_path: Optional[str] = None
        self.device: Optional[torch.device] = None
        self.model: Optional[MultiMediaSentenceModel] = None
        self.modelling_params: Optional[Dict[str, Any]] = None
        self.embeddings: Optional[List[torch.Tensor, torch.Tensor]] = None
        self.game_data: Optional[List[Dict[str, Any]]] = None
        self.id_name_map: Optional[Dict[int, Any]] = None

        self.config_path: Optional[str] = None
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

        self.config_path = os.path.join(model_dir, "config_sentence_v1.yaml")
        self.cached_data_folder = model_dir

        self.config = yaml.load(open(self.config_path), Loader=yaml.FullLoader)
        self.modelling_params = self.config["model"]
        self.data_path = self.config["data"]["dataset_path"]
        self.logger.info(f"[SENTENCE SERVICE] Data path : {self.data_path}")

        if os.path.exists(os.path.join(self.cached_data_folder, "game_data.pt")):
            self.game_data = torch.load(os.path.join(self.cached_data_folder, "game_data.pt"))
        else:
            self.game_data = get_metadata(self.data_path)

        self._initialize_from_cached()
        self._initialize_model()

        self.logger.info("Loaded metadata successfully")

        self.logger.info(f"Initialized {self.__class__.__name__} successfully")

        self.initialized = True

    def _initialize_model(self) -> None:
        self.model = MultiMediaSentenceModel(model_config=self.modelling_params,
                                             k=self.n_similar)
        self.model.eval()
        cached_faiss_path = os.path.join(self.cached_data_folder, "sentence_model_faiss.index")
        self.logger.info(f"Loading model from {cached_faiss_path}. Existing: {os.path.exists(cached_faiss_path)}")
        self.model.initialize(self.game_data, cached_faiss_path=cached_faiss_path)
        self.logger.info("Loaded model successfully")

    def _initialize_from_cached(self) -> None:
        similarity_graph = torch.load(os.path.join(self.cached_data_folder, "similarity_graph.pt"))
        self.id_name_map = {k: v["game_name"] for k, v in similarity_graph.items()}
        self.logger.info("Loaded similarity graph successfully")

    def reinitialize(self) -> None:
        """

        """
        self.logger.info("Reinitializing model")
        if os.path.exists(os.path.join(self.cached_data_folder, "game_data.pt")):
            self.game_data = torch.load(os.path.join(self.cached_data_folder, "game_data.pt"))
        else:
            self.game_data = get_metadata(self.data_path)
            torch.save(self.game_data, os.path.join(self.cached_data_folder, "game_data.pt"))
        self._initialize_from_cached()

        self.model.initialize(self.game_data,
                              cached_faiss_path=os.path.join(self.cached_data_folder, "sentence_model_faiss.index"))

    def preprocess_text_data(self, text: str) -> str:
        """

        :param text:
        :return:
        """
        return text

    def preprocess(self, data: Any) -> Any:
        """

        :param data:
        :return:
        """
        self.logger.info(f"Received Data : {data}")
        for row in data:
            query = row.get("data") or row.get("body")
            self.logger.info(f"Received Query : {query}")
            search_type = query.get("search_type")
            search_input = query.get("search_input")

            if search_type != "text":
                raise ValueError(f"Invalid search type: {search_type}")
            search_input = str(search_input)
            return self.preprocess_text_data(search_input)

    def _search_game_with_text(self, query_sentence: str) -> List[Dict[str, Any]]:
        similar_games = self.model.predict_most_similar(query_sentence)
        return [
            {
                "id": game_id,
                "name": self.id_name_map[game_id],
            }
            for game_id, _ in similar_games
        ]

    def inference(self, preprocessed_inputs: Union[str, int], **kwargs
                  ) -> Union[List[Dict[str, Any]], List[int]]:
        """

        :param preprocessed_inputs:
        :param kwargs:
        :return:
        """
        if isinstance(preprocessed_inputs, str):
            return self._search_game_with_text(preprocessed_inputs)
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
