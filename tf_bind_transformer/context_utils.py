import torch
import logging
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging

logging.set_verbosity_error()

MODELS = dict(
    pubmed = dict(
        dim = 768,
        path = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    )
)

@torch.no_grad()
def tokenize_texts(
    texts,
    *,
    device,
    max_length = 256,
    model_name = 'pubmed',
    hidden_state_index = -1,
    return_cls_token = True,
):
    assert model_name in MODELS, f'{model_name} not found in available text transformers to use'

    if isinstance(texts, str):
        texts = [texts]

    path = MODELS[model_name]['path']
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForMaskedLM.from_pretrained(path)
    encoding = tokenizer.batch_encode_plus(texts, add_special_tokens = True, padding = True, truncation = True, max_length = max_length, return_attention_mask = True, return_tensors = "pt")

    outputs = model(**encoding, output_hidden_states = True)
    hidden_states = outputs.hidden_states[hidden_state_index]

    if return_cls_token:
        return hidden_states[:, 0].to(device)

    return hidden_states.mean(dim = 1).to(device)
