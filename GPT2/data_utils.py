from typing import List, Dict, Any
from pathlib import Path
import json

from torch import Tensor, LongTensor, stack, argmax
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollatorMixin
from miditok import MIDITokenizer
from miditoolkit import MidiFile
from tqdm import tqdm



class MIDIDataset(Dataset):
    r"""Dataset for generator training

    :param files_paths: list of paths to files to load.
    :param tokenizer: tokenizer object, to use to load MIDIs instead of tokens. (default: None)
    """

    def __init__(self, files_paths: List[Path], min_seq_len: int, max_seq_len: int, tokenizer: MIDITokenizer = None):
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

        self.samples = samples

    def __getitem__(self, idx) -> Dict[str, LongTensor]:
        return {"input_ids": self.samples[idx], "labels": self.samples[idx]}

    def __len__(self) -> int: return len(self.samples)

    def __repr__(self): return self.__str__()

    def __str__(self) -> str: return 'No data loaded' if len(self) == 0 else f'{len(self.samples)} samples'


def _pad_batch(examples: List[Dict[str, LongTensor]], pad_token: int) -> LongTensor:
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""

    length_of_first = examples[0]["input_ids"].size(0)

    # Check if padding is necessary.
    are_tensors_same_length = all(x["input_ids"].size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return stack([e["input_ids"] for e in examples], dim=0).long()

    # Creating the full tensor and filling it with our data.
    return pad_sequence([e["input_ids"] for e in examples], batch_first=True, padding_value=pad_token).long()


class DataCollatorGen(DataCollatorMixin):
    def __init__(self, pad_token: int, return_tensors: str = "pt"):
        """Collator that simply pad the input sequences.
        Input_ids will be padded with the pad token given, while labels will be
        padded with -100.

        :param pad_token: pas token
        :param return_tensors:
        """
        self.pad_token = pad_token
        self.return_tensors = return_tensors

    def __call__(self, batch: List[Dict[str, Any]], return_tensors=None) -> Dict[str, LongTensor]:
        x, y = _pad_batch(batch, self.pad_token), _pad_batch(batch, -100)
        return {"input_ids": x, "labels": y}  # will be shifted in GPT2LMHead forward


