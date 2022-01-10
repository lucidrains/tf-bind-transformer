import torch
import esm
from torch.nn.utils.rnn import pad_sequence
from tf_bind_transformer.cache_utils import cache_fn, run_once

GLOBAL_VARIABLES = dict(model = None)

ESM_EMBED_DIM = 1280

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

@run_once('init_esm')
def init_esm():
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    GLOBAL_VARIABLES['model'] = (model, batch_converter)

def get_single_esm_repr(protein_str):
    init_esm()
    model, batch_converter = GLOBAL_VARIABLES['model']

    data = [('protein', protein_str)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])

    token_representations = results['representations'][33]
    representation = token_representations[0][1 : len(protein_str) + 1]
    return representation

def get_esm_repr(proteins, device):
    if isinstance(proteins, torch.Tensor):
        proteins = tensor_to_aa_str(proteins)

    get_protein_repr_fn = cache_fn(get_single_esm_repr, path = 'esm/proteins')

    representations = []
    for subunits in proteins:
        subunits = (subunits,) if not isinstance(subunits, tuple) else subunits
        subunits_representations = list(map(get_protein_repr_fn, subunits))
        subunits_representations = torch.cat(subunits_representations, dim = 0)
        representations.append(subunits_representations)

    lengths = [seq_repr.shape[0] for seq_repr in representations]
    masks = torch.arange(max(lengths), device = device)[None, :] <  torch.tensor(lengths, device = device)[:, None]
    padded_representations = pad_sequence(representations, batch_first = True)

    return padded_representations.to(device), masks.to(device)

# for fetching transcription factor sequences

GENE_IDENTIFIER_MAP = {
    'RXR': 'RXRA'
}

NAMES_WITH_HYPHENS = {
    'NKX3-1',
    'NKX2-1',
    'NKX2-5',
    'SS18-SSX'
}

def parse_gene_name(name):
    if '-' not in name or name in NAMES_WITH_HYPHENS:
        name = GENE_IDENTIFIER_MAP.get(name, name)
        return (name,)

    first, *rest = name.split('-')

    parsed_rest = []

    for name in rest:
        if len(name) == 1:
            name = f'{first[:-1]}{name}'
        parsed_rest.append(name)

    return tuple([first, *parsed_rest])
