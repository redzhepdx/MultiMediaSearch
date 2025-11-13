import argparse
import os.path
from typing import Any, Dict, List, Tuple, Union

import albumentations as A
import gdown
import torch
import yaml
from colorlog import logger
from torch import nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, PreTrainedTokenizer

from multimedia_search.rule_engine.engine import MetadataRuleEngine
from multimedia_search.semantic_search_engine.data.dataset import MultiMediaDataset
from multimedia_search.semantic_search_engine.embedding_ops import get_embedding_pairs, get_faiss_index, \
    get_image_embeddings, \
    get_pairwise_fast, get_text_embeddings
from multimedia_search.semantic_search_engine.modelling.multi_domain_model import MultiMediaCLIPModel
from multimedia_search.semantic_search_engine.modelling.sentence_model import MultiMediaSentenceModel
from multimedia_search.semantic_search_engine.pipelines.utils import state_dict_from_disk
from multimedia_search.utility.data_ops import create_dataset, get_metadata
from multimedia_search.utility.get_data import retrieve_from_api


def get_data_from_url(file_name: str) -> None:
    url = "https://api.gamegator.net/v1/ai/products"
    retrieve_from_api(url=url, file_name=file_name)


def create_dataset_from_data_file(data_file_path: str, output_folder: str) -> None:
    create_dataset(data_file_path=data_file_path, output_folder=output_folder)


def download_trained_model(drive_url: str, output: str) -> None:
    """
    Download trained model from google drive
    :param drive_url: path to the model in google drive
    :param output: path to save the model
    :return: None
    """
    gdown.download(drive_url, output, quiet=False)


def prepare_multi_model(config: Dict[str, Any],
                        model_path: str
                        ) -> Tuple[str, str, nn.Module, DataLoader, MultiMediaDataset, PreTrainedTokenizer]:
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
    if modelling_params["resume_from_checkpoint"]:
        # Read pytorch lightning checkpoint
        state_dict = state_dict_from_disk(modelling_params["resume_from_checkpoint"],
                                          rename_in_layers={"model.image_encoder.": "image_encoder.",
                                                            "model.text_encoder.": "text_encoder.",
                                                            "model.video_encoder.": "video_encoder.",
                                                            "model.image_projection.": "image_projection.",
                                                            "model.text_projection.": "text_projection.",
                                                            "model.video_projection.": "video_projection."})
        model.load_state_dict(state_dict)
    else:
        # Read custom pytorch checkpoint
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return data_path, device, model, test_data_loader, test_dataset, tokenizer


def get_annotations(data_path: str) -> List[Dict[str, Any]]:
    annotations = get_metadata(data_path)
    torch.save(annotations, "cached_data_new/game_data.pt")
    return annotations


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
        if os.path.exists("cached_data_new/test_embeddings_image.pt"):
            test_embeddings = torch.load("cached_data_new/test_embeddings_image.pt")
        else:
            test_embeddings = get_image_embeddings(model, test_data_loader, device=device)
            torch.save(test_embeddings, "cached_data_new/test_embeddings_image.pt")
    elif search_in == "text":
        if not os.path.exists("cached_data_new/test_embeddings_text.pt"):
            test_embeddings = get_text_embeddings(model, test_data_loader, device=device)
            torch.save(test_embeddings, "cached_data_new/test_embeddings_text.pt")
        else:
            test_embeddings = torch.load("cached_data_new/test_embeddings_text.pt")
    elif not os.path.exists("cached_data_new/test_embeddings_mixed.pt"):
        test_embeddings = get_embedding_pairs(model, test_data_loader, device=device)
        torch.save(test_embeddings, "cached_data_new/test_embeddings_mixed.pt")
    else:
        test_embeddings = torch.load("cached_data_new/test_embeddings_mixed.pt")
    return test_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Create dataset from GameGator API.")
    parser.add_argument("--config_file", type=str, default=None, help="Path to the config file.", required=True)
    parser.add_argument("--dataset_folder", type=str, default=None, help="Path to the output folder.", required=True)
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model.", required=False)
    return parser.parse_args()


def main():
    args = parse_args()

    # Read config file
    logger.info(f"Reading config file from {args.config_file}\n")
    config = yaml.load(open(args.config_file), Loader=yaml.FullLoader)
    os.makedirs("cached_data_new", exist_ok=True)

    logger.info("Creating dataset from GameGator API.\n")
    # Download api data from url
    get_data_from_url(file_name="data_new.json")

    logger.info("Creating dataset from GameGator API.\n")
    # Build a dataset from the retrieved data
    create_dataset_from_data_file(data_file_path="data_new.json", output_folder=args.dataset_folder)

    logger.info("Updating the config file.\n")
    config["data"]["dataset_path"] = args.dataset_folder

    logger.info("Preparing the data for the semantic search engine.\n")
    # Get all the annotations and save them to a file
    annotations = get_annotations(data_path=args.dataset_folder)
    print(f"Len annotations : {len(annotations)}")

    # If the model path is not provided, download the model from google drive
    if args.model_path is None:
        logger.info("Downloading the model from google drive.\n")
        model_url = "https://drive.google.com/file/d/1NhwRQr_x6LdAXt5kve5-H2oLvkmz1u65/view?usp=sharing"
        model_path = "trained_models/best_v2.pt"
        download_trained_model(drive_url=model_url,
                               output=model_path)
    else:
        logger.info("Using the provided model.\n")
        model_path = args.model_path

    logger.info("Preparing the model and rule engine.\n")
    # Prepare the model and rule engine
    rule_engine = MetadataRuleEngine(annotations=annotations)
    _, device, model, test_data_loader, _, _ = prepare_multi_model(config=config, model_path=model_path)

    # Prepare the embeddings and the faiss indexes
    for search_in in ["image", "text", "mixed"]:
        logger.info(f"Preparing the embeddings for {search_in} search.\n")
        test_embeddings = prepare_embeddings(model=model, test_data_loader=test_data_loader, search_in=search_in,
                                             device=device)

        if search_in in ["image", "text"]:
            logger.info(f"Preparing the faiss index for {search_in} search.\n")
            faiss_index_path = f"cached_data_new/faiss_vector_{search_in}.index"
            get_faiss_index(faiss_index_path=faiss_index_path, test_embeddings=test_embeddings)

    logger.info("Preparing the pairwise similarity scores.\n")
    # Get the pairwise similarity scores
    if os.path.exists("cached_data_new/similarity_graph.pt"):
        similarity_graph = torch.load("cached_data_new/similarity_graph.pt")
    else:
        similarity_parameters = config["similarity_parameters"]
        similarity_graph = get_pairwise_fast(test_embeddings=test_embeddings, annotations=annotations,
                                             rule_engine=rule_engine, similarity_parameters=similarity_parameters)

        torch.save(similarity_graph, "cached_data_new/similarity_graph.pt")

    logger.info("Preparing the sentence model.\n")
    # Prepare the sentence model
    sentence_model = MultiMediaSentenceModel(model_config={'model_name_or_path': 'all-MiniLM-L6-v2'})

    logger.info("Initializing the sentence model to save the faiss index and labels.\n")
    # Initialize the sentence model to save the faiss index and labels
    sentence_model.initialize(annotations, cached_faiss_path="cached_data_new/sentence_model_faiss.index")


if __name__ == '__main__':
    main()
