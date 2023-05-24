from pathlib import Path
import json
import os
import wandb

from torch import Tensor, argmax
from torchtoolkit.data import create_subsets
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments
from evaluate import load as load_metric

from data_utils import MIDIDataset, DataCollatorGen


if __name__ == "__main__":

    # Set up wandb for tracking
    os.environ["WANDB_PROJECT"] = "gpt-2"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    wandb.init(project="gpt-2",
               entity="jlenzy")

    # Loads tokens and create data loaders for training
    tokens_paths = list(Path('tokenized_data').glob("**/*.json"))
    dataset = MIDIDataset(
        tokens_paths, max_seq_len=512, min_seq_len=384,
    )
    subset_train, subset_valid = create_subsets(dataset, [0.3])


    with open("tokenizer_params/vocab.json", 'r') as fp:
        vocab = json.load(fp)
    print(f"length of vocab: {len(vocab)}")

    # Instantiate model
    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=2048,
        n_embd=512,
        n_layer=8,
        n_head=8,
        n_inner=2048,
        resid_pdrop=.1,
        embd_pdrop=.1,
        attn_pdrop=.1,
        padding_token_id=vocab['PAD_None'],
        bos_token_id=vocab['BOS_None'],
        eos_token_id=vocab['EOS_None'],
    )
    print("#---- Loading the Model")
    print(f"Vocab size: {len(vocab)}")
    model = GPT2LMHeadModel(config)

    # Train
    metrics = {metric: load_metric(metric) for metric in ["accuracy"]}

    def compute_metrics(eval_pred):
        """Computes metrics for pretraining.
        Must use proprocess_logits function that converts logits to predictions (argmax or sampling).

        :param eval_pred: EvalPrediction containing predictions and labels
        :return: metrics
        """
        predictions, labels = eval_pred
        not_pad_mask = labels != -100
        labels, predictions = labels[not_pad_mask], predictions[not_pad_mask]
        return metrics["accuracy"].compute(predictions=predictions.flatten(), references=labels.flatten())


    def preprocess_logits(logits: Tensor, _: Tensor) -> Tensor:
        """Preprocesses the logits before accumulating them during evaluation.
        This allows to significantly reduce the memory usage and make the training tractable.
        """
        pred_ids = argmax(logits, dim=-1)  # long dtype
        return pred_ids

    training_config = TrainingArguments(
        "runs", False, True, True, False, "steps",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=48,
        gradient_accumulation_steps=3,
        eval_accumulation_steps=None,
        eval_steps=1000,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_grad_norm=3.0,
        max_steps=100000,
        lr_scheduler_type="cosine_with_restarts",
        warmup_ratio=0.3,
        log_level="debug",
        logging_strategy="steps",
        logging_steps=20,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=5,
        no_cuda=False,
        seed=444,
        fp16=False, #Set to true for GPU
        load_best_model_at_end=True,
        label_smoothing_factor=0.,
        optim="adamw_torch",
        report_to=["wandb"],
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_config,
        data_collator=DataCollatorGen(vocab["PAD_None"]),
        train_dataset=subset_train,
        eval_dataset=subset_valid,
        compute_metrics=compute_metrics,
        callbacks=None,
        preprocess_logits_for_metrics=preprocess_logits,
    )

    # Training
    print("#---- Starting Training")
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()