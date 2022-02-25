from dotenv import load_dotenv

# set path to cache in .env and unset the next comment
# load_dotenv()

from enformer_pytorch import Enformer
from tf_bind_transformer import AdapterModel, BigWigTrainer

# training constants

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 1e-4   # Deepmind used 1e-4 for fine-tuning of Enformer

# effective batch size of BATCH_SIZE * GRAD_ACCUM_STEPS = 16

VALIDATE_EVERY = 250
GRAD_CLIP_MAX_NORM = 1.5

TFACTOR_FOLDER = './tfactor.fastas'
HUMAN_FASTA_FILE_PATH = './hg38.ml.fa'
MOUSE_FASTA_FILE_PATH = './mm10.ml.fa'

HUMAN_LOCI_PATH = './chip_atlas/human_sequences.bed'
MOUSE_LOCI_PATH = './chip_atlas/mouse_sequences.bed'
BIGWIG_PATH = './chip_atlas/bigwig'
ANNOT_FILE_PATH =  './chip_atlas/annot.tab'

TARGET_LENGTH = 896

HELD_OUT_TARGET = ['GATA2']

# instantiate enformer or load pretrained

enformer = Enformer.from_pretrained('EleutherAI/enformer-preview', target_length = TARGET_LENGTH)

# instantiate model wrapper that takes in enformer

model = AdapterModel(
    enformer = enformer,
    use_aa_embeds = True,
    use_free_text_context = True,
    free_text_embed_method = 'mean_pool',
    aa_embed_encoder = 'protalbert',
    use_corr_coef_loss = True               # use 1 - pearson_corr_coef loss
).cuda()


# trainer class for fine-tuning

trainer = BigWigTrainer(
    model,
    human_loci_path = HUMAN_LOCI_PATH,
    mouse_loci_path = MOUSE_LOCI_PATH,
    human_fasta_file = HUMAN_FASTA_FILE_PATH,
    mouse_fasta_file = MOUSE_FASTA_FILE_PATH,
    bigwig_folder_path = BIGWIG_PATH,
    annot_file_path = ANNOT_FILE_PATH,
    target_length = TARGET_LENGTH,
    lr = LEARNING_RATE,
    batch_size = BATCH_SIZE,
    shuffle = True,
    validate_every = VALIDATE_EVERY,
    grad_clip_norm = GRAD_CLIP_MAX_NORM,
    grad_accum_every = GRAD_ACCUM_STEPS,
    human_factor_fasta_folder = TFACTOR_FOLDER,
    mouse_factor_fasta_folder = TFACTOR_FOLDER,
    held_out_targets = HELD_OUT_TARGET
)

# do gradient steps in a while loop

while True:
    _ = trainer()
