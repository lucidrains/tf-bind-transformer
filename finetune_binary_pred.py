from dotenv import load_dotenv

# set path to cache in .env and unset the next comment
# load_dotenv()

from enformer_pytorch import Enformer
from tf_bind_transformer import AdapterModel, Trainer

# instantiate enformer or load pretrained

enformer = Enformer.from_hparams(
    dim = 768,
    depth = 4,
    heads = 8,
    target_length = -1,
    use_convnext = True,
    num_downsamples = 6   # resolution of 2 ^ 6 == 64bp
)

# instantiate model wrapper that takes in enformer

model = AdapterModel(
    enformer = enformer,
    use_aa_embeds = True,
    use_free_text_context = True,
    free_text_embed_method = 'mean_pool',
    binary_target = True,
    target_mse_loss = False,
    use_squeeze_excite = True,
    aa_embed_encoder = 'protalbert'
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

CONTEXT_LENGTH = 4096

SCOPED_NEGS_REMAP_PATH = './neg-npy/remap2022.bed'
SCOPED_NEGS_PATH = './neg-npy'

TRAIN_CHROMOSOMES = [*range(1, 24, 2), 'X'] # train on odd chromosomes
VALID_CHROMOSOMES = [*range(2, 24, 2)]      # validate on even

HELD_OUT_TARGET = ['AFF4']

# trainer class for fine-tuning

trainer = Trainer(
    model,
    context_length = CONTEXT_LENGTH,
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
    held_out_targets = HELD_OUT_TARGET,
    include_scoped_negs = True,
    scoped_negs_remap_bed_path = SCOPED_NEGS_REMAP_PATH,
    scoped_negs_path = SCOPED_NEGS_PATH,
)

# do gradient steps in a while loop

while True:
    _ = trainer(finetune_enformer_ln_only = False)
