from typing import List, Tuple, Dict, Callable, Any, Union
from pathlib import Path
import json

from torch import Tensor, LongTensor, stack, flip, cat, full, argmax

from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader
from torchtoolkit.data import create_subsets
from miditok import MIDILike, MIDITokenizer
from miditoolkit import MidiFile
from tqdm import tqdm

from data_utils import pad_data


class MIDIDataset(Dataset):
    r"""Dataset for generator training

    :param files_paths: list of paths to files to load.
    :param tokenizer: tokenizer object, to use to load MIDIs instead of tokens. (default: None)
    """

    def __init__(self, files_paths: List[Path], min_seq_len: int, max_seq_len: int, padding_value=0, tokenizer: MIDITokenizer = None):
        samples = []

        for file_path in tqdm(files_paths, desc=f'Loading data: {files_paths[0].parent}'):
            if file_path.suffix in ["mid", "midi", "MID", "MIDI"]:
                midi = MidiFile(file_path)
                for _ in range(len(midi.instruments) - 1):
                    del midi.instruments[1]  # removes all tracks except first one
                tokens = tokenizer.midi_to_tokens(midi)[0].ids
            else:
                with open(file_path) as json_file:
                    tokens = json.load(json_file)['ids'][0]  # first track
            i = 0
            while i < len(tokens):
                if i >= len(tokens) - min_seq_len:
                    break  # last sample is too short
                samples.append(LongTensor(tokens[i:i + max_seq_len]))
                i += len(samples[-1])  # could be replaced with max_seq_len

        self.samples, self.masks = pad_data(samples, max_seq_len, padding_value=padding_value)

    def __getitem__(self, idx):
        return self.samples[idx], self.masks[idx]

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return 'No data loaded' if len(self) == 0 else f'{len(self.samples)} samples'



def tokenize_dataset(midi_filepath, tokenization_type, pitch_range,
                     beat_res, nb_velocities, additional_tokens,
                     special_tokens, max_seq_len, min_seq_len,
                     pad_dataset=True):

    assert tokenization_type in ["MIDILike"], "Invalid tokenization type specified"
    assert max_seq_len > min_seq_len

    if tokenization_type == "MIDILike":
        tokenizer = MIDILike(pitch_range, beat_res, nb_velocities, additional_tokens, special_tokens=special_tokens)

    else:
        print("Unable to tokenize")
        pass

    midi_paths = list(Path(midi_filepath).glob('**/*.mid')) + list(Path(midi_filepath).glob('**/*.midi'))
    # Create empty directory for the tokens
    tokens_path = Path('tokens')
    for filepath in tokens_path.glob('**/*'):
        if filepath.is_file():
            filepath.unlink()

    tokenizer.tokenize_midi_dataset(midi_paths, tokens_path, apply_bpe=False)


    # Loads tokens and create data loaders for training
    tokens_paths = list(Path(tokens_path).glob("**/*.json"))
    dataset = MIDIDataset(
        tokens_paths, max_seq_len=max_seq_len, min_seq_len=min_seq_len
    )

    return dataset, tokenizer

    # subset_train, subset_valid = create_subsets(dataset, [0.3])
    #
    # return subset_train, subset_valid

def create_pad_collate(sequence_length, padding_value=0):
    def pad_collate(batch):
        data = [item for item in batch]
        data = pad_sequence(data, batch_first=True, padding_value=padding_value)
        if data.size(1) < sequence_length:
            data = pad(data, (0, sequence_length - data.size(1)))
        else:
            data = data[:, :sequence_length]
        return data
    return pad_collate


