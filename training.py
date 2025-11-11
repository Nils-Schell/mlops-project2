import os
import argparse
from datetime import datetime
from typing import Optional

import datasets
import evaluate
import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from dotenv import load_dotenv
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class GLUEDataModule(L.LightningDataModule):
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
            self,
            model_name_or_path: str,
            task_name: str = "mrpc",
            max_seq_length: int = 128,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, padding="max_length", truncation=True
        )
        features["labels"] = example_batch["label"]
        return features


class GLUETransformer(L.LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            num_labels: int,
            task_name: str,
            learning_rate: float = 2e-5,
            warmup_ratio: float = 0.0,
            beta1: float = 0.9,
            gradient_clipping_val: float = 0.0,
            weight_decay: float = 0.0,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            eval_splits: Optional[list] = None,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = evaluate.load(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )
        self.validation_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        self.validation_step_outputs.append({"loss": val_loss, "preds": preds, "labels": labels})
        return val_loss

    def on_validation_epoch_end(self):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(self.validation_step_outputs):
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            self.validation_step_outputs.clear()
            return loss

        preds = torch.cat([x["preds"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(self.hparams.beta1, 0.999)
        )

        if self.hparams.warmup_ratio > 0 and hasattr(self.trainer, 'estimated_stepping_batches'):
            num_training_steps = self.trainer.estimated_stepping_batches
            num_warmup_steps = int(num_training_steps * self.hparams.warmup_ratio)
        else:
            num_warmup_steps = 0
            if self.hparams.warmup_ratio > 0:
                print("Warning: Trainer.estimated_stepping_batches not available. Warmup steps set to 0.")

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches if hasattr(self.trainer,
                                                                                  'estimated_stepping_batches') else 1000,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


def main():
    load_dotenv()
    if not os.getenv("WANDB_API_KEY"):
        print("WARNING: WANDB_API_KEY not found in .env file.")

    parser = argparse.ArgumentParser(description="Train a GLUE Transformer model.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size. (NEU)")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.2, help="Warmup ratio.")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1 parameter. (NEU)")
    parser.add_argument("--checkpoint_dir", type=str, default="models", help="Directory to save local checkpoints.")

    args = parser.parse_args()

    EPOCHS = 3
    MODEL_NAME = "distilbert-base-uncased"
    TASK_NAME = "mrpc"
    MAX_SEQ_LENGTH = 128
    GRAD_CLIP_VAL = 1.0

    L.seed_everything(42)

    run_name = (
        f"P2-{TASK_NAME}-lr_{args.learning_rate}-bs_{args.train_batch_size}-wd_{args.weight_decay}"
        f"-wr_{args.warmup_ratio}-b1_{args.beta1}-ebs_{args.eval_batch_size}"
    )

    wandb_logger = WandbLogger(
        project="mlops-project2-glue-finetuning",
        entity="nils-schell-hslu",
        name=run_name,
        log_model="all",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f"{run_name}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        verbose=True
    )

    dm = GLUEDataModule(
        model_name_or_path=MODEL_NAME,
        task_name=TASK_NAME,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    dm.setup("fit")

    model = GLUETransformer(
        model_name_or_path=MODEL_NAME,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=TASK_NAME,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        beta1=args.beta1,
        gradient_clipping_val=GRAD_CLIP_VAL,
        train_batch_size=dm.train_batch_size,
        eval_batch_size=dm.eval_batch_size,
    )

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        gradient_clip_val=GRAD_CLIP_VAL,
        callbacks=[checkpoint_callback],
    )

    device_name = str(trainer.strategy.root_device)
    print(f"\nTrainer initialized. Using device: {device_name}\n")

    print(f"Starting training: {run_name} (for {EPOCHS} epochs)")
    trainer.fit(model, datamodule=dm)

    print("Training complete.")


if __name__ == "__main__":
    main()