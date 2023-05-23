from pathlib import Path
import json
import argparse
import wandb
import os

from torchtoolkit.data import create_subsets
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments

from evaluate import load as load_metric
from miditok import MIDILike
from miditok.constants import CHORD_MAPS

from data import MIDIDataset, DataCollatorGen, preprocess_logits, compute_metrics

# Training setting variables



# parser = argparse.ArgumentParser()
# parser.add_argument("--wandb", type=bool, help="log to wandb", default=True)
# parser.add_argument("--wandb_project", type=str, help="WANDB Project Name", default="GPT2")


if __name__ == "__main__":

    # Set up wandb for tracking
    os.environ["WANDB_PROJECT"] = "gpt-2"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    wandb.init(project="gpt-2",
               entity="jlenzy")

    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_path", type=str, help="Path to MIDI Files")

    args = parser.parse_args()
    MIDI_PATH = args.midi_path

    #MIDI_PATH = "/Users/jlenz/Desktop/Datasets/maestro-mini"

    # Our parameters
    pitch_range = range(21, 109)
    beat_res = {(0, 4): 8, (4, 12): 4}
    nb_velocities = 32
    additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True,
                         'rest_range': (2, 8),  # (half, 8 beats)
                         'nb_tempos': 32,  # nb of tempo bins
                         'tempo_range': (40, 250),  # (min, max)
                         'Program': False,
                         "chord_maps": CHORD_MAPS,
                         "chord_tokens_with_root_note": True,
                         "chord_unknown": False}
    special_tokens = ["PAD", "BOS", "EOS"]

    # Creates the tokenizer convert MIDIs to tokens
    print("#---- Tokenizing the data")
    tokens_path = Path('Maestro_tokens_no_bpe')
    tokenizer = MIDILike(pitch_range, beat_res, nb_velocities, additional_tokens, special_tokens=special_tokens)
    midi_paths = list(Path(MIDI_PATH).glob('**/*.mid')) + list(Path(MIDI_PATH).glob('**/*.midi'))
    print(f"Training on {len(midi_paths)} MIDI files.\n")
    tokenizer.tokenize_midi_dataset(midi_paths, tokens_path)

    # Save tokenization settings
    tokenizer_params = Path('tokenizer_params')
    if not os.path.exists(tokenizer_params):
        os.makedirs(tokenizer_params)

    with open(os.path.join(tokenizer_params, "vocab.json"), "w") as fp:
        json.dump(tokenizer.vocab, fp)
    tokenizer.save_params(out_path="tokenizer_params/config.json")

    # Loads tokens and create data loaders for training
    tokens_paths = list(Path('Maestro_tokens_no_bpe').glob("**/*.json"))
    dataset = MIDIDataset(
        tokens_paths, max_seq_len=512, min_seq_len=384,
    )
    subset_train, subset_valid = create_subsets(dataset, [0.3])

    # Instantiate model
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=2048,
        n_embd=512,
        n_layer=8,
        n_head=8,
        n_inner=2048,
        resid_pdrop=.1,
        embd_pdrop=.1,
        attn_pdrop=.1,
        padding_token_id=tokenizer['PAD_None'],
        bos_token_id=tokenizer['BOS_None'],
        eos_token_id=tokenizer['EOS_None'],
    )
    print("#---- Loading the Model")
    model = GPT2LMHeadModel(config)

    # Train
    metrics = {metric: load_metric(metric) for metric in ["accuracy"]}

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
        data_collator=DataCollatorGen(tokenizer["PAD_None"]),
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