import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from multimedia_search.semantic_search_engine.pipelines.clip_pipeline import ClipPipeline
from multimedia_search.utility.general import read_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", "--config", type=str, required=True, help="Path to the config file")
    return parser.parse_args()


def main():
    args = get_args()
    print(args.config)

    # train model with pytorch lightning
    config = read_config(config_path=args.config)
    pipeline = ClipPipeline(config=config)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config["logging"]["checkpoint_dir"],
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    wandb_logger = WandbLogger(
        name=config["logging"]["wandb"]["experiment_name"],
        project=config["logging"]["wandb"]["project_name"],
        log_model=True,
        save_dir=config["logging"]["wandb"]["save_dir"],
        version=config["logging"]["wandb"]["version"]
    )

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=config["hyper-parameters"]["num_epochs"],
        progress_bar_refresh_rate=1,
        num_sanity_val_steps=2,
        sync_batchnorm=True,
        callbacks=[checkpoint_callback],
        benchmark=True,
        log_every_n_steps=10,
        logger=wandb_logger
    )

    trainer.fit(pipeline)


if __name__ == "__main__":
    main()
