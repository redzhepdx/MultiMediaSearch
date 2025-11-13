import os
from typing import Any, Dict, Optional

import albumentations as A
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, DistilBertTokenizer

from multimedia_search.semantic_search_engine.data.dataset import MultiMediaDataset
from multimedia_search.semantic_search_engine.modelling.multi_domain_model import MultiMediaCLIPModel
from multimedia_search.semantic_search_engine.pipelines.utils import state_dict_from_disk


class ClipPipeline(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super(ClipPipeline, self).__init__()
        self.config = config
        self.hyperparams = config["hyper-parameters"]
        self.modelling_params = config["model"]
        self.data_path = config["data"]["dataset_path"]
        self.custom_local_save_ckpts = False

        self.use_image_encoder = self.modelling_params.get("use_image_encoder")
        if isinstance(self.use_image_encoder, str):
            self.use_image_encoder = eval(self.use_image_encoder)

        self.use_video_encoder = self.modelling_params.get("use_video_encoder")
        if isinstance(self.use_video_encoder, str):
            self.use_video_encoder = eval(self.use_video_encoder)

        self.image_transforms: Optional[A.Compose] = None
        self.video_transforms: Optional[AutoImageProcessor] = None
        self.tokenizer: Optional[DistilBertTokenizer] = None
        self.model: Optional[MultiMediaCLIPModel] = None
        self.optimizers: Optional[torch.optim.Optimizer] = None

        self.test_dataset: Optional[MultiMediaDataset] = None

        self._setup_preprocessors()
        self._setup_model()

    def _setup_preprocessors(self):
        if self.use_image_encoder:
            self.image_transforms = A.Compose([
                    A.PadIfNeeded(min_height=self.modelling_params["image_size"],
                                  min_width=self.modelling_params["image_size"], p=1.0),
                    A.Resize(self.modelling_params["image_size"], self.modelling_params["image_size"], p=1.0),
                    A.Normalize(p=1.0)]
            )

        if self.use_video_encoder:
            self.video_transforms = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

        self.tokenizer = DistilBertTokenizer.from_pretrained(self.modelling_params["text_tokenizer"])

    def _setup_model(self):
        self.model = MultiMediaCLIPModel(model_config=self.modelling_params,
                                         hyperparams=self.hyperparams).to(self.device)
        if self.modelling_params.get("resume_from_checkpoint"):
            state_dict = state_dict_from_disk(self.modelling_params["resume_from_checkpoint"],
                                              rename_in_layers={"model.image_encoder.": "image_encoder.",
                                                                "model.text_encoder.": "text_encoder.",
                                                                "model.video_encoder.": "video_encoder.",
                                                                "model.image_projection.": "image_projection.",
                                                                "model.text_projection.": "text_projection.",
                                                                "model.video_projection.": "video_projection."}
                                              )
            self.model.load_state_dict(state_dict)

    def load_model_from_checkpoint(self, checkpoint_path: str, model: nn.Module) -> nn.Module:
        """
        :param checkpoint_path: path to the checkpoint
        :param model: model to be loaded
        :return: Trained model
        """
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        return model

    def train_dataloader(self) -> DataLoader:
        train_dataset = MultiMediaDataset(images_folder=os.path.join(self.data_path, "images"),
                                          annotations_folder=os.path.join(self.data_path, "annotations"),
                                          img_preprocessor=self.image_transforms,
                                          text_tokenizer=self.tokenizer,
                                          videos_folder=os.path.join(self.data_path, "videos"),
                                          video_processor=self.video_transforms,
                                          max_length=self.modelling_params["max_length"],
                                          use_images=self.use_image_encoder,
                                          use_videos=self.use_video_encoder
                                          )

        return DataLoader(train_dataset, batch_size=self.hyperparams["batch_size"], shuffle=True,
                          num_workers=self.hyperparams["num_workers"], pin_memory=True, drop_last=True)

    def test_dataloader(self) -> DataLoader:
        test_dataset = MultiMediaDataset(images_folder=os.path.join(self.data_path, "images"),
                                         annotations_folder=os.path.join(self.data_path, "annotations"),
                                         img_preprocessor=self.image_transforms,
                                         text_tokenizer=self.tokenizer,
                                         videos_folder=os.path.join(self.data_path, "videos"),
                                         video_processor=self.video_transforms,
                                         max_length=self.modelling_params["max_length"],
                                         use_images=self.use_image_encoder,
                                         use_videos=self.use_video_encoder)
        return DataLoader(test_dataset, batch_size=1, shuffle=False,
                          num_workers=self.hyperparams["num_workers"], pin_memory=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        test_dataset = MultiMediaDataset(images_folder=os.path.join(self.data_path, "images"),
                                         annotations_folder=os.path.join(self.data_path, "annotations"),
                                         img_preprocessor=self.image_transforms,
                                         text_tokenizer=self.tokenizer,
                                         videos_folder=os.path.join(self.data_path, "videos"),
                                         video_processor=self.video_transforms,
                                         max_length=self.modelling_params["max_length"],
                                         use_images=self.use_image_encoder,
                                         use_videos=self.use_video_encoder)
        return DataLoader(test_dataset, batch_size=self.hyperparams["batch_size"], shuffle=False,
                          num_workers=self.hyperparams["num_workers"], pin_memory=True, drop_last=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        params = [
                {"params": self.model.text_encoder.parameters(), "lr": float(self.hyperparams["text_encoder_lr"])},
                {"params": self.model.text_projection.parameters(), "lr": float(self.hyperparams["head_lr"]),
                 "weight_decay": float(self.hyperparams["weight_decay"])},
        ]
        if self.use_image_encoder:
            params.append(
                    {"params": self.model.image_encoder.parameters(),
                     "lr": float(self.hyperparams["image_encoder_lr"])})
            params.append({"params": self.model.image_projection.parameters(), "lr": float(self.hyperparams["head_lr"]),
                           "weight_decay": float(self.hyperparams["weight_decay"])})

        if self.use_video_encoder:
            params.append(
                    {"params": self.model.video_encoder.parameters(),
                     "lr": float(self.hyperparams["video_encoder_lr"])})
            params.append({"params": self.model.video_projection.parameters(), "lr": float(self.hyperparams["head_lr"]),
                           "weight_decay": float(self.hyperparams["weight_decay"])})

        optimizer = torch.optim.AdamW(params, weight_decay=0.)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=self.hyperparams["scheduler_patience"],
                factor=self.hyperparams["scheduler_factor"],
        )

        self.optimizers = optimizer
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_loss"}

    def training_step(self, batch: Dict, batch_idx: int) -> Dict[str, Any]:
        loss = self.forward(batch)

        self.log("val_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict[str, Any]:
        loss = self.forward(batch)

        self.log("train_loss", loss)
        return {"loss": loss}

    def test_step(self, batch: Dict, batch_idx: int) -> Dict[str, Any]:
        loss = self.forward(batch)

        self.log("test_loss", loss)
        return {"loss": loss}

    def on_train_end(self) -> None:
        """
        :return: None
        """
        # reload the best model
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        if best_model_path:
            state_dict = state_dict_from_disk(best_model_path,
                                              rename_in_layers={"model.image_encoder.": "image_encoder.",
                                                                "model.text_encoder.": "text_encoder.",
                                                                "model.video_encoder.": "video_encoder.",
                                                                "model.image_projection.": "image_projection.",
                                                                "model.text_projection.": "text_projection.",
                                                                "model.video_projection.": "video_projection."}
                                              )
            self.model.load_state_dict(state_dict)

    def on_validation_epoch_end(self, save=torch.save) -> None:
        """
        :return: None
        """
        if self.custom_local_save_ckpts:
            save(self.model.state_dict(),
                 os.path.join(self.model_checkpoint_path, f"best_multi_media_{self.epoch + 1}.pt"))

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """

        :param batch:
        :return:
        """
        video_data = batch.get("video")

        batch = {k: v.to(self.device) for k, v in batch.items() if k not in ["summary", "video"]}

        if self.use_video_encoder:
            video_data = {k: v.to(self.device) for k, v in video_data.items() if isinstance(v, torch.Tensor)}
            batch["video"] = video_data

        return self.model(batch)
