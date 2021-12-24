import torch
import esm
from torch.nn.utils.rnn import pad_sequence

INT_TO_AA_STR_MAP = {
    0: 'A',
    1: 'C',
    2: 'D',
    3: 'E',
    4: 'F',
    5: 'G',
    6: 'H',
    7: 'I',
    8: 'K',
    9: 'L',
    10: 'M',
    11: 'N',
    12: 'P',
    13: 'Q',
    14: 'R',
    15: 'S',
    16: 'T',
    17: 'V',
    18: 'W',
    19: 'Y',
    20: '_'
}

def tensor_to_aa_str(t):
    str_seqs = []
    for int_seq in t.unbind(dim = 0):
        str_seq = list(map(lambda t: INT_TO_AA_STR_MAP[t] if t != 20 else '', int_seq.tolist()))
        str_seqs.append(''.join(str_seq))
    return str_seqs

def init_esm():
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

def get_esm_repr(
    aa,
    model,
    batch_converter,
    return_padded_with_masks = False
):
    device = aa.device
    data = [(f'protein{ind}', aa_str) for ind, aa_str in enumerate(tensor_to_aa_str(aa))]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
    token_representations = results["representations"][33]

    sequence_representations = []
    for i, (_, seq) in enumerate(data):
        token_repr = token_representations[i, 1 : len(seq) + 1]
        sequence_representations.append(token_repr.to(device))

    if not return_padded_with_masks:
        return sequence_representations

    lengths = [seq_repr.shape[0] for seq_repr in sequence_representations]
    masks = torch.arange(max(lengths))[None, :] <  torch.tensor(lengths)[:, None]
    padded_sequences = pad_sequence(sequence_representations, batch_first = True)
    return padded_sequences, masks
