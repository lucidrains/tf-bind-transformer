import torch
from enformer_pytorch import load_pretrained_model

from tf_bind_transformer import HyperTransformerAdapterModel

from tf_bind_transformer.training_utils import get_optimizer
from tf_bind_transformer.data import RemapAllPeakDataset, NegativePeakDataset, get_dataloader, collate_dl_outputs

# constants

BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8

# effective batch size of BATCH_SIZE * GRAD_ACCUM_STEPS = 16

VALIDATE_EVERY = 250
GRAD_CLIP_MAX_NORM = 1.5

REMAP_FILE_PATH = './remap2022_all.bed'
TFACTOR_FOLDER = './tfactor.fastas'
FASTA_FILE_PATH = './hg38.ml.fa'
NON_PEAK_PATH = './generated-non-peaks.bed'

TRAIN_CHROMOSOMES = [*range(1, 24, 2), 'X'] # train on odd chromosomes
VALID_CHROMOSOMES = [*range(2, 24, 2)]      # validate on even

HELD_OUT_TARGET = ['SOX2']

# datasets and dataloaders

ds = RemapAllPeakDataset(
    bed_file = REMAP_FILE_PATH,                      # path to remap bed file
    fasta_file = FASTA_FILE_PATH,                    # fasta file (human)
    factor_fasta_folder = TFACTOR_FOLDER,            # path to downloaded tfactors fastas
    filter_chromosome_ids = TRAIN_CHROMOSOMES,       # even chromosomes for training
    exclude_targets = HELD_OUT_TARGET,               # hold out certain targets for validation
    context_length = 4096                            # context length to be fetched
)

neg_ds = NegativePeakDataset(
    negative_bed_file = NON_PEAK_PATH,               # path to negative peaks generated with script above
    remap_bed_file = REMAP_FILE_PATH,                # path to remap bed file
    fasta_file = FASTA_FILE_PATH,                    # fasta file (human)
    factor_fasta_folder = TFACTOR_FOLDER,            # path to downloaded tfactors fastas
    filter_chromosome_ids = TRAIN_CHROMOSOMES,       # even chromosomes for training
    exclude_targets = HELD_OUT_TARGET,               # hold out certain targets for validation
    context_length = 4096                            # context length to be fetched
)

valid_ds = RemapAllPeakDataset(
    bed_file = REMAP_FILE_PATH,
    fasta_file = FASTA_FILE_PATH,
    factor_fasta_folder = TFACTOR_FOLDER,
    filter_chromosome_ids = VALID_CHROMOSOMES,       # odd chromosomes for validation
    include_targets = HELD_OUT_TARGET,
    context_length = 4096
)

valid_neg_ds = NegativePeakDataset(
    negative_bed_file = NON_PEAK_PATH,               # path to negative peaks generated with script above
    remap_bed_file = REMAP_FILE_PATH,
    fasta_file = FASTA_FILE_PATH,
    factor_fasta_folder = TFACTOR_FOLDER,
    filter_chromosome_ids = VALID_CHROMOSOMES,       # odd chromosomes for validation
    include_targets = HELD_OUT_TARGET,
    context_length = 4096
)

dl = get_dataloader(ds, cycle_iter = True, shuffle = True, batch_size = BATCH_SIZE)
neg_dl = get_dataloader(neg_ds, cycle_iter = True, shuffle = True, batch_size = BATCH_SIZE)

valid_dl = get_dataloader(valid_ds, cycle_iter = True, batch_size = BATCH_SIZE)
valid_neg_dl = get_dataloader(valid_neg_ds, cycle_iter = True, batch_size = BATCH_SIZE)

# instantiate enformer or load pretrained

enformer = load_pretrained_model('preview', target_length = -1)

# instantiate model wrapper that takes in enformer

model = HyperTransformerAdapterModel(
    enformer = enformer,
    use_esm_embeds = True,
    use_free_text_context = True,
    free_text_embed_method = 'mean_pool',
    binary_target = True,
    target_mse_loss = False,
    use_squeeze_excite = True
).cuda()

optim = get_optimizer(model.parameters())

i = 0
while True:
    model.train()

    total_loss = 0.
    for _ in range(GRAD_ACCUM_STEPS):
        # data

        seq, tf_aa, contextual_texts, _, binary_target = collate_dl_outputs(next(dl), next(neg_dl))
        seq, binary_target = seq.cuda(), binary_target.cuda()

        # train

        loss = model(
            seq,
            target = binary_target,
            aa = tf_aa,
            contextual_free_text = contextual_texts,
            finetune_enformer_ln_only = True
        )

        total_loss += loss.item()
        (loss / GRAD_ACCUM_STEPS).backward()

    print(f'loss: {total_loss / GRAD_ACCUM_STEPS}')

    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
    optim.step()
    optim.zero_grad()

    if not (i % VALIDATE_EVERY):
        model.eval()

        seq, tf_aa, contextual_texts, _, binary_target = collate_dl_outputs(next(valid_dl), next(valid_neg_dl))
        seq, binary_target = seq.cuda(), binary_target.cuda()

        valid_logits = model(
            seq,
            aa = tf_aa,
            contextual_free_text = contextual_texts
        )

        valid_loss = model.loss_fn(valid_logits, binary_target.float())
        valid_accuracy = ((valid_logits.sigmoid() > 0.5).int() == binary_target).sum() / (binary_target.numel())

        print(f'valid loss: {valid_loss.item()}')
        print(f'valid accuracy: {valid_accuracy.item()}')

    i += 1
