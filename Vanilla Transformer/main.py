
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from data import tokenize_dataset, create_pad_collate
from data_utils import pad_data
from pathlib import Path
from model import TransformerModel


if __name__ == "__main__":

    midi_filepath = "/Users/jlenz/Desktop/Qosmo/datasets/maestro-v3.0.0"
    midi_filepath_small = Path("/Users/jlenz/Desktop/Qosmo/datasets/maestro-v3.0.0/mini")
    # Our parameters
    pitch_range = range(21, 109)
    beat_res = {(0, 4): 8, (4, 12): 4}
    nb_velocities = 32
    additional_tokens = {'Chord': False, 'Rest': False, 'Tempo': False, 'Program': False, 'TimeSignature': False,
                         'rest_range': (2, 8),  # (half, 8 beats)
                         'nb_tempos': 32,  # nb of tempo bins
                         'tempo_range': (40, 250)}  # (min, max)
    special_tokens = ["PAD", "BOS", "EOS", "MASK"]


    dataset, tokenizer = tokenize_dataset(midi_filepath=midi_filepath_small, tokenization_type="MIDILike", pitch_range=pitch_range,
                             beat_res=beat_res, nb_velocities=nb_velocities, additional_tokens=additional_tokens,
                             special_tokens=special_tokens, max_seq_len=512, min_seq_len=64)


    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in dataloader:
        inputs, masks = batch
        print(inputs.shape)
        print(masks.shape)
        break




