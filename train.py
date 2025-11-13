import argparse
import os.path
from typing import Any

import albumentations as A
import torch
import yaml
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, DistilBertTokenizer

from multimedia_search.semantic_search_engine.data.dataset import MultiMediaDataset
from multimedia_search.semantic_search_engine.modelling.multi_domain_model import MultiMediaCLIPModel
from multimedia_search.semantic_search_engine.modelling.utils import AvgMeter, get_lr


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: Optimizer, lr_scheduler: Any, step: str,
                device: str = "cuda", use_image: bool = False, use_video: bool = False) -> AvgMeter:
    """

    :param model:
    :param train_loader:
    :param optimizer:
    :param lr_scheduler:
    :param step:
    :param device:
    :param use_image:
    :param use_video:
    :return:
    """
    assert use_image or use_video, "At least one of the image or video encoders must be used"
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        video_data = batch.get("video")

        batch = {k: v.to(device) for k, v in batch.items() if k not in ["summary", "video"]}
        batch_size = batch.get("input_ids").size(0)

        if use_video:
            video_data = {k: v.to(device) for k, v in video_data.items() if isinstance(v, torch.Tensor)}
            batch["video"] = video_data

        loss = model(batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == "batch":
            lr_scheduler.step()

        loss_meter.update(loss.item(), batch_size)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model: nn.Module, valid_loader: DataLoader, device: str, use_image: bool = False,
                use_video: bool = False) -> AvgMeter:
    """

    :param model:
    :param valid_loader:
    :param device:
    :param use_image:
    :param use_video:
    :return:
    """
    assert use_image or use_video, "At least one of the image or video encoders must be used"
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        video_data = batch.get("video")
        batch = {k: v.to(device) for k, v in batch.items() if k not in ["summary", "video"]}

        if use_video:
            video_data = {k: v.to(device) for k, v in video_data.items() if isinstance(v, torch.Tensor)}
            batch["video"] = video_data

        batch_size = batch.get("input_ids").size(0)

        loss = model(batch)

        loss_meter.update(loss.item(), batch_size)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(config_path: str) -> None:
    """

    :param config_path:
    :return:
    """
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    hyperparams = config["hyper-parameters"]
    modelling_params = config["model"]
    data_path = config["data"]["dataset_path"]

    use_image_encoder = modelling_params.get("use_image_encoder")
    if isinstance(use_image_encoder, str):
        use_image_encoder = eval(use_image_encoder)

    use_video_encoder = modelling_params.get("use_video_encoder")
    if isinstance(use_video_encoder, str):
        use_video_encoder = eval(use_video_encoder)

    image_transforms = A.Compose([
        A.PadIfNeeded(min_height=modelling_params["image_size"], min_width=modelling_params["image_size"], p=1.0),
        A.Resize(modelling_params["image_size"], modelling_params["image_size"], p=1.0),
        A.Normalize(p=1.0)])

    video_transforms = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

    tokenizer = DistilBertTokenizer.from_pretrained(modelling_params["text_tokenizer"])

    train_dataset = MultiMediaDataset(images_folder=os.path.join(data_path, "images"),
                                      annotations_folder=os.path.join(data_path, "annotations"),
                                      img_preprocessor=image_transforms,
                                      text_tokenizer=tokenizer,
                                      # videos_folder=os.path.join(data_path, "videos"),
                                      # video_processor=video_transforms,
                                      max_length=modelling_params["max_length"],
                                      use_images=use_image_encoder,
                                      use_videos=use_video_encoder
                                      )

    test_dataset = MultiMediaDataset(images_folder=os.path.join(data_path, "images"),
                                     annotations_folder=os.path.join(data_path, "annotations"),
                                     img_preprocessor=image_transforms,
                                     text_tokenizer=tokenizer,
                                     # videos_folder=os.path.join(data_path, "videos"),
                                     # video_processor=video_transforms,
                                     max_length=modelling_params["max_length"],
                                     use_images=use_image_encoder,
                                     use_videos=use_video_encoder)

    train_data_loader = DataLoader(
        train_dataset, batch_size=hyperparams["batch_size"], shuffle=True, num_workers=hyperparams["num_workers"]
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=hyperparams["batch_size"], shuffle=False, num_workers=hyperparams["num_workers"]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiMediaCLIPModel(model_config=modelling_params, hyperparams=hyperparams).to(device)

    params = [
        {"params": model.text_encoder.parameters(), "lr": float(hyperparams["text_encoder_lr"])},
        {"params": model.text_projection.parameters(), "lr": float(hyperparams["head_lr"]),
         "weight_decay": float(hyperparams["weight_decay"])},
    ]
    if use_image_encoder:
        params.append({"params": model.image_encoder.parameters(), "lr": float(hyperparams["image_encoder_lr"])})
        params.append({"params": model.image_projection.parameters(), "lr": float(hyperparams["head_lr"]),
                       "weight_decay": float(hyperparams["weight_decay"])})

    if use_video_encoder:
        params.append({"params": model.video_encoder.parameters(), "lr": float(hyperparams["video_encoder_lr"])})
        params.append({"params": model.video_projection.parameters(), "lr": float(hyperparams["head_lr"]),
                       "weight_decay": float(hyperparams["weight_decay"])})

    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=hyperparams["scheduler_patience"], factor=hyperparams["scheduler_factor"],
    )

    step = "epoch"

    best_loss = float('inf')
    for epoch in range(hyperparams["num_epochs"]):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_data_loader, optimizer, lr_scheduler, step, device,
                                 use_image=use_image_encoder, use_video=use_video_encoder)
        print(f"Train Loss: {train_loss.avg}")
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, test_data_loader, device=device,
                                     use_image=use_image_encoder, use_video=use_video_encoder)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
        torch.save(model.state_dict(), f"best_multi_media_{epoch + 1}.pt")
        print("Saved Best Model!")

        lr_scheduler.step(valid_loss.avg)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", "--config", type=str, required=True, help="Path to the config file")
    return parser.parse_args()


def main():
    args = get_args()
    train(config_path=args.config)


if __name__ == "__main__":
    main()
