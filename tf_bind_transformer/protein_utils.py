import torch
import os
import re
import esm
from torch.nn.utils.rnn import pad_sequence
from transformers import AlbertTokenizer, AutoModelForMaskedLM, logging
from tf_bind_transformer.cache_utils import cache_fn, run_once

def exists(val):
    return val is not None

def map_values(fn, dictionary):
    return {k: fn(v) for k, v in dictionary.items()}

PROTEIN_EMBED_USE_CPU = os.getenv('PROTEIN_EMBED_USE_CPU', None) is not None

if PROTEIN_EMBED_USE_CPU:
    print('calculating protein embed only on cpu')

GLOBAL_VARIABLES = dict(model = None)

ESM_MAX_LENGTH = 1024
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

    if not PROTEIN_EMBED_USE_CPU:
        model = model.cuda()

    GLOBAL_VARIABLES['model'] = (model, batch_converter)

def get_single_esm_repr(protein_str):
    init_esm()
    model, batch_converter = GLOBAL_VARIABLES['model']

    data = [('protein', protein_str)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    if batch_tokens.shape[1] > ESM_MAX_LENGTH:
        print(f'warning max length protein esm: {protein_str}')

    batch_tokens = batch_tokens[:, :ESM_MAX_LENGTH]

    if not PROTEIN_EMBED_USE_CPU:
        batch_tokens = batch_tokens.cuda()

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

# get protein dim

def get_protein_embed_dim(model_name):
    if model_name == 'esm':
        return ESM_EMBED_DIM
    elif model_name == 'prot_albert':
        return PROT_ALBERT_DIM
    else:
        raise ValueError(f'{model_name} not in supported models')

# prot-albert 2048 context length

PROT_ALBERT_PATH = 'Rostlab/prot_albert'
PROT_ALBERT_DIM = 4096
PROT_ALBERT_MAX_LENGTH = 2048

GLOBAL_VARIABLES = dict(tokenizer = None, model = None)

def protein_str_with_spaces(protein_str):
    protein_str = re.sub(r"[UZOB]", 'X', protein_str)
    return ' '.join([*protein_str])

@run_once('init_prot_albert')
def init_prot_albert():
    GLOBAL_VARIABLES['tokenizer'] = AlbertTokenizer.from_pretrained(PROT_ALBERT_PATH, do_lower_case = False)
    model = AutoModelForMaskedLM.from_pretrained(PROT_ALBERT_PATH)

    if not PROTEIN_EMBED_USE_CPU:
        model = model.cuda()

    GLOBAL_VARIABLES['model'] = model

def get_single_prot_albert_repr(
    protein_str,
    max_length = PROT_ALBERT_MAX_LENGTH,
    hidden_state_index = -1
):
    init_prot_albert()
    model = GLOBAL_VARIABLES['model']
    tokenizer = GLOBAL_VARIABLES['tokenizer']

    encoding = tokenizer.batch_encode_plus(
        [protein_str_with_spaces(protein_str)],
        add_special_tokens = True,
        padding = True,
        truncation = True,
        max_length = max_length,
        return_attention_mask = True,
        return_tensors = 'pt'
    )

    if not PROTEIN_EMBED_USE_CPU:
        encoding = map_values(lambda t: t.cuda(), encoding)

    outputs = model(**encoding, output_hidden_states = True)
    hidden_state = outputs.hidden_states[hidden_state_index][0]
    return hidden_state

def get_prot_albert_repr(
    proteins,
    device,
    max_length = PROT_ALBERT_MAX_LENGTH,
    hidden_state_index = -1
):
    if isinstance(proteins, str):
        proteins = [proteins]

    get_protein_repr_fn = cache_fn(get_single_prot_albert_repr, path = f'proteins/prot_albert')

    representations = [get_protein_repr_fn(protein, max_length = max_length, hidden_state_index = hidden_state_index) for protein in proteins]

    lengths = [seq_repr.shape[0] for seq_repr in representations]
    masks = torch.arange(max(lengths), device = device)[None, :] <  torch.tensor(lengths, device = device)[:, None]
    padded_representations = pad_sequence(representations, batch_first = True)

    return padded_representations.to(device), masks.to(device)
