from enformer_pytorch import load_pretrained_model
from tf_bind_transformer import AttentionAdapterModel, Trainer

# instantiate enformer or load pretrained

enformer = load_pretrained_model('preview', target_length = -1)

# instantiate model wrapper that takes in enformer

model = AttentionAdapterModel(
    enformer = enformer,
    use_aa_embeds = True,
    use_free_text_context = True,
    free_text_embed_method = 'mean_pool',
    binary_target = True,
    target_mse_loss = False,
    use_squeeze_excite = True
).cuda()


# training constants

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

HELD_OUT_TARGET = ['AFF4']

# trainer class for fine-tuning

trainer = Trainer(
    model,
    batch_size = BATCH_SIZE,
    validate_every = VALIDATE_EVERY,
    grad_clip_norm = GRAD_CLIP_MAX_NORM,
    grad_accum_every = GRAD_ACCUM_STEPS,
    remap_bed_file = REMAP_FILE_PATH,
    negative_bed_file = NON_PEAK_PATH,
    factor_fasta_folder = TFACTOR_FOLDER,
    fasta_file = FASTA_FILE_PATH,
    train_chromosome_ids = TRAIN_CHROMOSOMES,
    valid_chromosome_ids = VALID_CHROMOSOMES,
    held_out_targets = HELD_OUT_TARGET
)

# do gradient steps in a while loop

while True:
    _ = trainer()
