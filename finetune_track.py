from dotenv import load_dotenv

# set path to cache in .env and unset the next comment
# load_dotenv()

from enformer_pytorch import load_pretrained_model
from tf_bind_transformer import AdapterModel, BigWigTrainer

# training constants

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8

# effective batch size of BATCH_SIZE * GRAD_ACCUM_STEPS = 16

VALIDATE_EVERY = 250
GRAD_CLIP_MAX_NORM = 1.5

TFACTOR_FOLDER = './tfactor.fastas'
HUMAN_FASTA_FILE_PATH = './hg38.ml.fa'
MOUSE_FASTA_FILE_PATH = './mm10.ml.fa'

LOCI_PATH = './sequences.bed'
BIGWIG_PATH = './bigwig_folder'
ANNOT_FILE_PATH =  './experiments.tab'
TARGET_LENGTH = 896

TRAIN_CHROMOSOMES = [*range(1, 24, 2), 'X'] # train on odd chromosomes
VALID_CHROMOSOMES = [*range(2, 24, 2)]      # validate on even

HELD_OUT_TARGET = ['SOX2']

# instantiate enformer or load pretrained

enformer = load_pretrained_model('preview', target_length = TARGET_LENGTH)

# instantiate model wrapper that takes in enformer

model = AdapterModel(
    enformer = enformer,
    use_aa_embeds = True,
    use_free_text_context = True,
    free_text_embed_method = 'mean_pool',
    aa_embed_encoder = 'protalbert'
).cuda()


# trainer class for fine-tuning

trainer = BigWigTrainer(
    model,
    loci_path = LOCI_PATH,
    bigwig_folder_path = BIGWIG_PATH,
    annot_file_path = ANNOT_FILE_PATH,
    target_length = TARGET_LENGTH,
    batch_size = BATCH_SIZE,
    validate_every = VALIDATE_EVERY,
    grad_clip_norm = GRAD_CLIP_MAX_NORM,
    grad_accum_every = GRAD_ACCUM_STEPS,
    factor_fasta_folder = TFACTOR_FOLDER,
    human_fasta_file = HUMAN_FASTA_FILE_PATH,
    mouse_fasta_file = MOUSE_FASTA_FILE_PATH,
    train_chromosome_ids = TRAIN_CHROMOSOMES,
    valid_chromosome_ids = VALID_CHROMOSOMES,
    held_out_targets = HELD_OUT_TARGET
)

# do gradient steps in a while loop

while True:
    _ = trainer()
