import os

MODEL_NAME = "gamegator-mm-search"
MODEL_STORE = "model_store_mm"

FILES = [
    "../custom_configs/config_v1.yaml",
    "../trained_models/best.pt",
    "../cached_data/game_data.pt",
    "../cached_data/test_embeddings_text.pt",
    "../cached_data/test_embeddings_image.pt",
    "../cached_data/test_embeddings_mixed.pt",
    "../cached_data/pairwise_similarities.pt",
    "../cached_data/similarity_graph.pt",
    "../cached_data/faiss_vector_text.index",
    "../cached_data/faiss_vector_image.index"
]


def create_mar_file():
    files_text = ",".join(FILES)
    print(f"Creating MAR file with files: {files_text}")
    os.system(
        f"torch-model-archiver --model-name {MODEL_NAME} --extra-files {files_text} "
        f"--handler endpoint_multimodel.py --version 0.1")
    os.makedirs(MODEL_STORE, exist_ok=True)
    os.system(f"mv {MODEL_NAME}.mar {MODEL_STORE}")
    print(f"Created MAR file at {MODEL_STORE}/{MODEL_NAME}.mar")


def start_service():
    os.system(f"torchserve --start --model-store {MODEL_STORE} --models {MODEL_NAME}.mar")
    print(f"Started service with model {MODEL_NAME}.mar")


def main():
    create_mar_file()
    start_service()


if __name__ == '__main__':
    main()
