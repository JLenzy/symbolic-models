import torch
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence

def pad_data(inputs, sequence_length, padding_value=0):
    """
    Given a list of 2D tensors, return them padded along the 0 dimension to specified length
    :param inputs: a list of 2D tensors
    :param sequence_length: (int) the total length
    :param padding_value: (optional, int) the specified value for padding
    :return: padded_inputs (list of 2D tensors), src_padding_masks
    """
    padded_inputs = []
    src_padding_masks = []

    for tensor in inputs:
        tensor_pad = torch.full((sequence_length,), padding_value, dtype=tensor.dtype)
        tensor_pad[:tensor.shape[0]] = tensor

        # Create a mask for this tensor
        mask = torch.full((sequence_length,), False, dtype=torch.bool)
        mask[:tensor.shape[0]] = True

        padded_inputs.append(tensor_pad)
        src_padding_masks.append(mask)

    return padded_inputs, src_padding_masks


# Source mask is used to identify where the padding idx are within the input sequence
def create_src_mask(sequence, padding_index):
    return (torch.tensor(sequence) != padding_index).unsqueeze(-2)

# target mask prevents the decoder from attending 'ahead' of current position during training
def create_tgt_mask(size):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def make_predictions(logits):
    # Reduces dim by 1, i.e. [Batch, Len, Predictions] -> [Batch, Len]
    return torch.argmax(logits, dim=-1)

