from typing import Any, Dict, Union

import timm
import torch
import torch.nn.functional as F
from torch import nn
from transformers import DistilBertConfig, DistilBertModel
from transformers.models.videomae import VideoMAEModel


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(self, model_config: Dict[str, Any]) -> None:
        super().__init__()
        self.model = timm.create_model(
                model_config["model_name"], model_config["pretrained"], num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = model_config["trainable_encoder_and_tokenizer"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class VideoEncoder(nn.Module):
    """
    Encode videos to a fixed size vector
    """

    def __init__(self, model_config: Dict[str, Any]) -> None:
        super().__init__()
        # create pre-trained VideoMAE model
        self.model = VideoMAEModel.from_pretrained(model_config["video_encoder_model"])
        for p in self.model.parameters():
            p.requires_grad = model_config["trainable_encoder_and_tokenizer"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        if x.dim() == 6:
            x = x.squeeze(1)
        outputs = self.model(x)
        return outputs.last_hidden_state[:, 0, :]


class TextEncoder(nn.Module):
    """

    :param model_config:
    """

    def __init__(self, model_config: Dict[str, Any]) -> None:
        super().__init__()
        if model_config["trainable_encoder_and_tokenizer"]:
            self.model = DistilBertModel.from_pretrained(model_config["text_encoder_model"])
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = model_config["trainable_encoder_and_tokenizer"]

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """

        :param input_ids:
        :param attention_mask:
        :return:
        """
        output = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    """

    :param embedding_dim:
    :param projection_dim:
    :param dropout:
    """

    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class MultiMediaCLIPModel(nn.Module):
    """

    :param model_config:
    :param hyperparams:
    """

    def __init__(self, model_config: Dict[str, Any], hyperparams: Dict[str, Any]) -> None:
        super().__init__()
        self.text_encoder = TextEncoder(model_config=model_config)

        self.text_projection = ProjectionHead(embedding_dim=model_config["text_embedding"],
                                              projection_dim=model_config["projection_dim"])

        use_image_encoder = model_config.get("use_image_encoder")
        if isinstance(use_image_encoder, str):
            use_image_encoder = eval(use_image_encoder)
        self.use_video_encoder = use_image_encoder
        if use_image_encoder:
            self.image_encoder = ImageEncoder(model_config=model_config)
            self.image_projection = ProjectionHead(embedding_dim=model_config["image_embedding"],
                                                   projection_dim=model_config["projection_dim"])

        use_video_encoder = model_config.get("use_video_encoder")
        if isinstance(use_video_encoder, str):
            use_video_encoder = eval(use_video_encoder)
        self.use_video_encoder = use_video_encoder
        if self.use_video_encoder:
            self.video_encoder = VideoEncoder(model_config=model_config)
            self.video_projection = ProjectionHead(embedding_dim=model_config["video_embedding"],
                                                   projection_dim=model_config["projection_dim"])

        self.temperature = hyperparams["temperature"]

    def joint_forward(self, batch: Dict[str, Union[torch.Tensor, Dict[str, Any]]]) -> torch.Tensor:
        """

        :param batch:
        :return:
        """
        text_features = self.text_encoder(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = self.text_projection(text_features)

        # Getting Video Features
        video_features = self.video_encoder(**batch["video"])
        video_embeddings = self.video_projection(video_features)

        # Getting Image Features
        image_features = self.image_encoder(batch["image"])
        image_embeddings = self.image_projection(image_features)

        # Calculating the similarities
        texts_similarity = text_embeddings @ text_embeddings.T
        video_similarity = video_embeddings @ video_embeddings.T
        images_similarity = image_embeddings @ image_embeddings.T

        # Calculating the loss for video and text
        video2text_logits = (text_embeddings @ video_embeddings.T) / self.temperature
        video2text_targets = F.softmax(
                (video_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        text2video_loss = cross_entropy(video2text_logits, video2text_targets, reduction='none')
        video2text_loss = cross_entropy(video2text_logits.T, video2text_targets.T, reduction='none')

        # Calculating the loss for image and text
        image2text_logits = (text_embeddings @ image_embeddings.T) / self.temperature
        image2text_targets = F.softmax(
                (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        text2image_loss = cross_entropy(image2text_logits, image2text_targets, reduction='none')
        image2text_loss = cross_entropy(image2text_logits.T, image2text_targets.T, reduction='none')

        # Calculating the loss for video and image
        image2video_logits = (video_embeddings @ image_embeddings.T) / self.temperature
        image2video_loss = cross_entropy(image2video_logits, image2text_targets, reduction='none')
        video2image_loss = cross_entropy(image2video_logits.T, image2text_targets.T, reduction='none')

        loss = (text2video_loss + video2text_loss + text2image_loss
                + image2text_loss + image2video_loss + video2image_loss) / 6.0

        return loss.mean()

    def video2text_forward(self, batch: Dict[str, Dict[str, Any]]) -> torch.Tensor:
        """

        :param batch:
        :return:
        """
        text_features = self.text_encoder(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = self.text_projection(text_features)
        texts_similarity = text_embeddings @ text_embeddings.T

        # Getting Video Features
        video_features = self.video_encoder(batch["video"]["pixel_values"])

        # Getting Image in the same dimension as text
        video_embeddings = self.video_projection(video_features)

        # Calculating the Loss
        video_logits = (text_embeddings @ video_embeddings.T) / self.temperature
        video_similarity = video_embeddings @ video_embeddings.T
        video_targets = F.softmax(
                (video_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(video_logits, video_targets, reduction='none')
        video_loss = cross_entropy(video_logits.T, video_targets.T, reduction='none')

        loss = (texts_loss + video_loss) / 2.0
        return loss.mean()

    def image2text_forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """

        :param batch:
        :return:
        """
        text_features = self.text_encoder(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = self.text_projection(text_features)
        texts_similarity = text_embeddings @ text_embeddings.T

        # Getting Image Features
        image_features = self.image_encoder(batch["image"])

        # Getting Image in the same dimension as text
        image_embeddings = self.image_projection(image_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        targets = F.softmax(
                (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')

        loss = (images_loss + texts_loss) / 2.0

        return loss.mean()

    def forward(self, batch: Dict[str, Union[torch.Tensor, Dict[str, Any]]],
                joint_training: bool = False) -> torch.Tensor:
        """

        :param batch:
        :param joint_training:
        :return:
        """
        if joint_training:
            return self.joint_forward(batch)

        if self.use_video_encoder:
            return self.video2text_forward(batch)

        return self.image2text_forward(batch)


def cross_entropy(preds: torch.Tensor, targets: torch.Tensor, reduction: str = 'none') -> torch.Tensor:
    """

    :param preds:
    :param targets:
    :param reduction:
    :return:
    """
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
