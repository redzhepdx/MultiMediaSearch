import os
from typing import Any, Dict, List, Optional, Tuple

import nltk
import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from torch import nn
from tqdm import tqdm

from multimedia_search.semantic_search_engine.embedding_ops import FaissKNeighbors


def get_data_from_annotations(annotations: List[Dict[str, Any]]) -> Tuple[List[str], List[int]]:
    nltk.download('punkt')
    sentences = []
    labels = []
    for annotation in tqdm(annotations):
        summary = annotation['product_summary']

        if summary is None:
            continue
        sentences.extend(sent_tokenize(summary))
        labels.extend([annotation['product_id']] * len(sent_tokenize(summary)))
    return sentences, labels


class MultiMediaSentenceModel(nn.Module):
    def __init__(self, model_config: Dict[str, Any], k: int = 10) -> None:
        super().__init__()
        self.model = SentenceTransformer(model_config['model_name_or_path'],
                                         device="cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.faiss_knn = FaissKNeighbors(k=k)

    def initialize(self, annotations: List[Dict[str, Any]],
                   cached_faiss_path: Optional[str] = "cached_data/sentence_model_faiss.index") -> None:
        """

        :param annotations:
        :param cached_faiss_path:
        :return:
        """
        print(os.path.exists(cached_faiss_path))
        if os.path.exists(cached_faiss_path):
            self.faiss_knn.load(cached_faiss_path)
        else:
            sentences, labels = get_data_from_annotations(annotations)
            self.fit(sentences, labels, save_path=cached_faiss_path)

    def encode(self, sentence: List[str]) -> np.ndarray:
        return self.model.encode(sentence)

    def fit(self, sentences: List[str], labels: List[int],
            save_path: str = "cached_data/sentence_model_faiss.index") -> None:
        sentence_embeddings = self.encode(sentences)
        labels = np.array(labels)
        self.faiss_knn.fit(sentence_embeddings, labels)
        self.faiss_knn.save(output_path=save_path)

    def predict(self, sentence: str) -> ndarray:
        sentence_embedding = self.encode([sentence])
        return self.faiss_knn.predict(sentence_embedding)

    def predict_most_similar(self, sentence: str) -> List[Tuple[Any, int]]:
        sentence_embedding = self.encode([sentence])
        return self.faiss_knn.predict_most_similar(sentence_embedding)
