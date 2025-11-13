import os

MODEL_NAME = "gamegator-ss-search"
MODEL_STORE = "model_store_ss"

FILES = [
    "../custom_configs/config_sentence_v1.yaml",
    "../cached_data/game_data.pt",
    "../cached_data/pairwise_similarities.pt",
    "../cached_data/similarity_graph.pt",
    "../cached_data/sentence_model_faiss.index",
    "../cached_data/sentence_labels.pt"

]


def create_mar_file():
    files_text = ",".join(FILES)
    print(f"Creating MAR file with files: {files_text}")
    os.system(
        f"torch-model-archiver --model-name {MODEL_NAME} --extra-files {files_text} "
        f"--handler endpoint_sentence.py --version 0.1")
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
