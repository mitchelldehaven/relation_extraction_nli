import argparse
import math
from functools import partial
from pathlib import Path
import json

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig

from src.paths import DATA_DIR, MODELS_DIR
from src.models.transformer import Transformer
from src.models.dataset import RelationExtractionDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tokenizer_max_length", type=int, default=128)
    parser.add_argument("--model_type", type=str, default="microsoft/deberta-large-mnli")
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--valid_interval", type=float, default=1.0)
    parser.add_argument("--use_random_negatives", action="store_true")
    parser.add_argument("--use_hard_negatives", action="store_true")
    parser.add_argument("--use_flipped_negatives", action="store_true")
    parser.add_argument("--use_flipped_random_negatives", action="store_true")
    parser.add_argument("--gradient_accumulations", default=1, type=int)
    args = parser.parse_args()
    return args


def train(args):
    if not args.use_random_negatives and not args.use_hard_negatives:
        print("Must use some form of negatives")
        exit(1)
    lr_callback = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_accuracy",
        dirpath=MODELS_DIR,
        filename=f"{args.model_type.replace('/', '')}" + "_{epoch:02d}-{valid_accuracy:.5f}",
        save_top_k=1,
        mode="max",
        verbose=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    config = AutoConfig.from_pretrained(args.model_type)
    train_data = []
    with open(DATA_DIR / "positive_train.json") as f:
        train_data.extend(json.load(f))
    valid_data = []
    with open(DATA_DIR / "positive_dev.json") as f:
        valid_data.extend(json.load(f))
    if args.use_random_negatives:
        with open(DATA_DIR / "random_negative_train.json") as f:
            train_data.extend(json.load(f))
        with open(DATA_DIR / "random_negative_dev.json") as f:
            valid_data.extend(json.load(f))
    if args.use_hard_negatives:
        with open(DATA_DIR / "hard_negative_train.json") as f:
            train_data.extend(json.load(f))
        with open(DATA_DIR / "hard_negative_dev.json") as f:
            valid_data.extend(json.load(f))
    if args.use_flipped_negatives:
        with open(DATA_DIR / "flipped_negative_train.json") as f:
            train_data.extend(json.load(f))
        with open(DATA_DIR / "flipped_negative_dev.json") as f:
            valid_data.extend(json.load(f))
    if args.use_flipped_random_negatives:
        with open(DATA_DIR / "flipped_random_negative_train.json") as f:
            train_data.extend(json.load(f))
        with open(DATA_DIR / "flipped_random_negative_dev.json") as f:
            valid_data.extend(json.load(f))      

    train_dataset = RelationExtractionDataset(train_data, config)
    valid_dataset = RelationExtractionDataset(valid_data, config)
    partial_collate_fn = partial(collate_fn, tokenizer=tokenizer, max_length=args.tokenizer_max_length)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=partial_collate_fn,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=2 * args.batch_size,
        num_workers=4,
        collate_fn=partial_collate_fn,
    )
    steps_per_epoch = math.ceil(
        (len(train_dataset) / args.batch_size) / args.gradient_accumulations
    )
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.epochs,
        default_root_dir="checkpoints",
        precision=16,
        callbacks=[lr_callback, checkpoint_callback],
        accumulate_grad_batches=args.gradient_accumulations,
        val_check_interval=args.valid_interval,
    )
    model = Transformer(args.model_type, epochs=args.epochs, steps_per_epoch=steps_per_epoch)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


if __name__ == "__main__":
    args = parse_args()
    train(args)
