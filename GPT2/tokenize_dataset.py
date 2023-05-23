from miditok import MIDILike

import argparse
import json
import os
import shutil
from pathlib import Path

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_path", type=str, help="Path to MIDI Files")
    args = parser.parse_args()
    MIDI_PATH = args.midi_path

    # Parameters
    pitch_range = range(21, 109)
    beat_res = {(0, 4): 8, (4, 12): 4}
    nb_velocities = 32
    additional_tokens = {'Chord': False,
                         'Rest': False,
                         'Tempo': False,
                         'Program': False,
                         'TimeSignature': False,
                         'rest_range': (2, 8),  # (half, 8 beats)
                         'nb_tempos': 32,  # nb of tempo bins
                         'tempo_range': (40, 250)}  # (min, max)
    special_tokens = ["PAD", "BOS", "EOS"]

    # Creates the tokenizer convert MIDIs to tokens
    print("#---- Tokenizing the data")
    tokens_path = Path('tokenized_data')

    # Check if the directory exists
    if tokens_path.exists() and tokens_path.is_dir():
        shutil.rmtree(tokens_path)
        os.makedirs(tokens_path)

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