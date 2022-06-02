## Transcription Factor binding predictions with Attention and Transformers

A repository with exploration into using transformers to predict DNA ↔ transcription factor binding.

## Install

Run the following at the project root to download dependencies

```bash
$ python setup.py install --user
```

Then you must install `pybedtools`  as well as `pyBigWig`

```bash
$ conda install --channel conda-forge --channel bioconda pybedtools pyBigWig
```

## Usage

```python
import torch
from tf_bind_transformer import AdapterModel

# instantiate enformer or load pretrained

from enformer_pytorch import Enformer
enformer = Enformer.from_hparams(
    dim = 1536,
    depth = 2,
    target_length = 256
)

# instantiate model wrapper that takes in enformer

model = AdapterModel(
    enformer = enformer,
    aa_embed_dim = 512,
    contextual_embed_dim = 256
).cuda()

# mock data

seq = torch.randint(0, 4, (1, 196_608 // 2)).cuda() # for ACGT

aa_embed = torch.randn(1, 1024, 512).cuda()
aa_mask = torch.ones(1, 1024).bool().cuda()

contextual_embed = torch.randn(1, 256).cuda() # contextual embeddings, including cell type, species, experimental parameter embeddings

target = torch.randn(1, 256).cuda()

# train

loss = model(
    seq,
    aa_embed = aa_embed,
    aa_mask = aa_mask,
    contextual_embed = contextual_embed,
    target = target
)

loss.backward()

# after a lot of training

corr_coef = model(
    seq,
    aa_embed = aa_embed,
    aa_mask = aa_mask,
    contextual_embed = contextual_embed,
    target = target,
    return_corr_coef = True
)
```

## Using ESM or ProtAlbert for fetching of transcription factor protein embeddings

```python
import torch
from enformer_pytorch import Enformer
from tf_bind_transformer import AdapterModel

enformer = Enformer.from_hparams(
    dim = 1536,
    depth = 2,
    target_length = 256
)

model = AdapterModel(
    enformer = enformer,
    use_aa_embeds = True,                            # set this to True
    aa_embed_encoder = 'esm',                        # by default, will use esm, but can be set to 'protalbert', which has a longer context length of 2048 (vs esm's 1024)
    contextual_embed_dim = 256
).cuda()

# mock data

seq = torch.randint(0, 4, (1, 196_608 // 2)).cuda()
tf_aa = torch.randint(0, 21, (1, 4)).cuda()           # transcription factor amino acid sequence, from 0 to 20

contextual_embed = torch.randn(1, 256).cuda()
target = torch.randn(1, 256).cuda()

# train

loss = model(
    seq,
    aa = tf_aa,
    contextual_embed = contextual_embed,
    target = target
)

loss.backward()
```

- [ ] add alphafold2

## Context passed in as free text

One can also pass the context (cell type, experimental parameters) directly as free text, which will be encoded by a text transformer trained on pubmed abstracts.

```python
import torch
from tf_bind_transformer import AdapterModel

# instantiate enformer or load pretrained

from enformer_pytorch import Enformer
enformer = Enformer.from_hparams(
    dim = 1536,
    depth = 2,
    target_length = 256
)

# instantiate model wrapper that takes in enformer

model = AdapterModel(
    enformer = enformer,
    use_aa_embeds = True,
    use_free_text_context = True,        # this must be set to True
    free_text_embed_method = 'mean_pool' # allow for mean pooling of embeddings, instead of using CLS token
).cuda()

# mock data

seq = torch.randint(0, 4, (2, 196_608 // 2)).cuda() # for ACGT
target = torch.randn(2, 256).cuda()

tf_aa = [
    'KVFGRCELAA',                  # single protein
    ('AMKRHGLDNY', 'YNDLGHRKMA')   # complex, representations will be concatted together
]

contextual_texts = [
    'cell type: GM12878 | dual cross-linked',
    'cell type: H1-hESC'
]

# train

loss = model(
    seq,
    target = target,
    aa = tf_aa,
    contextual_free_text = contextual_texts,
)

loss.backward()
```

## Binary prediction

For predicting binary outcome (bind or not bind), just set the `binary_targets = True` when initializing either adapters

ex.

```python
import torch
from tf_bind_transformer import AdapterModel
from enformer_pytorch import Enformer

# instantiate enformer or load pretrained

enformer = Enformer.from_hparams(
    dim = 1536,
    depth = 2,
    target_length = 256
)

# instantiate model wrapper that takes in enformer

model = AdapterModel(
    enformer = enformer,
    use_aa_embeds = True,
    use_free_text_context = True,
    free_text_embed_method = 'mean_pool',
    use_squeeze_excite = True,
    binary_target = True,                  # set this to True
    target_mse_loss = False                # whether to use MSE loss with target value
).cuda()

# mock data

seq = torch.randint(0, 4, (1, 196_608 // 2)).cuda() # for ACGT
binary_target = torch.randint(0, 2, (2,)).cuda()    # bind or not bind

tf_aa = [
    'KVFGRCELAA',
    ('AMKRHGLDNY', 'YNDLGHRKMA')
]

contextual_texts = [
    'cell type: GM12878 | chip-seq dual cross-linked',
    'cell type: H1-hESC | chip-seq single cross-linked'
]

# train

loss = model(
    seq,
    target = binary_target,
    aa = tf_aa,
    contextual_free_text = contextual_texts,
)

loss.backward()
```

## Predicting Tracks from BigWig

```python
from pathlib import Path
import torch
from enformer_pytorch import Enformer

from tf_bind_transformer import AdapterModel
from tf_bind_transformer.data_bigwig import BigWigDataset, get_bigwig_dataloader

# constants

ROOT = Path('.')
TFACTOR_TF = str(ROOT / 'tfactor.fastas')
ENFORMER_DATA = str(ROOT / 'chip_atlas' / 'sequences.bed')
FASTA_FILE_PATH = str(ROOT / 'hg38.ml.fa')
BIGWIG_PATH = str(ROOT / 'chip_atlas')
ANNOT_FILE_PATH = str(ROOT / 'chip_atlas' / 'annot.tab')

# bigwig dataset and dataloader

ds = BigWigDataset(
    factor_fasta_folder = TFACTOR_TF,
    bigwig_folder = BIGWIG_PATH,
    enformer_loci_path = ENFORMER_DATA,
    annot_file = ANNOT_FILE_PATH,
    fasta_file = FASTA_FILE_PATH
)

dl = get_bigwig_dataloader(ds, batch_size = 2)

# enformer

enformer = Enformer.from_hparams(
    dim = 384,
    depth = 1,
    target_length = 896
)

model = AdapterModel(
    enformer = enformer,
    use_aa_embeds = True,
    use_free_text_context = True
).cuda()

# mock data

seq, tf_aa, context_str, target = next(dl)
seq, target = seq.cuda(), target.cuda()

# train

loss = model(
    seq,
    aa = tf_aa,
    contextual_free_text = context_str,
    target = target
)

loss.backward()
```
## Data

The data needed for training is at <a href="https://remap.univ-amu.fr/download_page">this download page</a>.

### Transcription factors for Human and Mouse

To download the protein sequences for both species, you need to download the remap CRMs bed files, from which all the targets will be extracted, and fastas to be downloaded from Uniprot.

Download human remap CRMS

```bash
$ wget https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/remap2022_crm_macs2_hg38_v1_0.bed.gz
$ gzip -d remap2022_crm_macs2_hg38_v1_0.bed.gz
```

Download mouse remap CRMs

```bash
$ wget https://remap.univ-amu.fr/storage/remap2022/mm10/MACS2/remap2022_crm_macs2_mm10_v1_0.bed.gz
$ gzip -d remap2022_crm_macs2_mm10_v1_0.bed.gz
```

Downloading all human transcription factors

```bash
$ python script/fetch_factor_fastas.py --species human
```

For mouse transcription factors

```bash
$ python script/fetch_factor_fastas.py --species mouse
````

## Generating Negatives

### Generating Hard Negatives

For starters, the `RemapAllPeakDataset` will allow you to load data easily from the full remap peaks bed file for training.

Firstly you'll need to generate the non-peaks dataset by running the following function

```python
from tf_bind_transformer.data import generate_random_ranges_from_fasta

generate_random_ranges_from_fasta(
    './hg38.ml.fa',
    output_filename = './path/to/generated-non-peaks.bed',    # path to output file
    context_length = 4096,
    num_entries_per_key = 1_000_000,                          # number of negative samples
    filter_bed_files = [
        './remap_all.bed',                                    # filter out by all peak ranges (todo, allow filtering namespaced to experiment and target)
        './hg38.blacklist.rep.bed'                            # further filtering by blacklisted regions (gs://basenji_barnyard/hg38.blacklist.rep.bed)
    ]
)
```

### Generating Scoped Negatives - Negatives per Dataset (experiment + target + cell type)

Todo

## Simple Trainer class for fine-tuning

working fine-tuning training loop for bind / no-bind prediction

```python
import torch
from enformer_pytorch import Enformer

from tf_bind_transformer import AdapterModel, Trainer

# instantiate enformer or load pretrained

enformer = Enformer.from_pretrained('EleutherAI/enformer-official-rough', target_length = -1)

# instantiate model wrapper that takes in enformer

model = AdapterModel(
    enformer = enformer,
    use_aa_embeds = True,
    use_free_text_context = True,
    free_text_embed_method = 'mean_pool',
    binary_target = True,
    target_mse_loss = True,
    use_squeeze_excite = True,
    aux_read_value_loss = True     # use auxiliary read value loss, can be turned off
).cuda()

# pass the model (adapter + enformer) to the Trainer

trainer = Trainer(
    model,
    batch_size = 2,                                   # batch size
    context_length = 4096,                            # genetic sequence length
    grad_accum_every = 8,                             # gradient accumulation steps
    grad_clip_norm = 2.0,                             # gradient clipping
    validate_every = 250,
    remap_bed_file = './remap2022_all.bed',           # path to remap bed peaks
    negative_bed_file = './generated-non-peaks.bed',  # path to generated non-peaks
    factor_fasta_folder = './tfactor.fastas',         # path to factor fasta files
    fasta_file = './hg38.ml.fa',                      # human genome sequences
    train_chromosome_ids = [*range(1, 24, 2), 'X'],   # chromosomes to train on
    valid_chromosome_ids = [*range(2, 24, 2)],        # chromosomes to validate on
    held_out_targets = ['AFF4'],                      # targets to hold out for validation
    experiments_json_path = './data/experiments.json' # path to all experiments data, at this path relative to the project root, if repository is git cloned
)

while True:
    _ = trainer()

```

working fine-tuning script for training on new enformer tracks, with cross-attending transcription factor protein embeddings and cell type conditioning

```python
from dotenv import load_dotenv

# set path to cache in .env and unset the next comment
# load_dotenv()

from enformer_pytorch import Enformer
from tf_bind_transformer import AdapterModel, BigWigTrainer

# training constants

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8

# effective batch size of BATCH_SIZE * GRAD_ACCUM_STEPS = 16

VALIDATE_EVERY = 250
GRAD_CLIP_MAX_NORM = 1.5

TFACTOR_FOLDER = './tfactor.fastas'
FASTA_FILE_PATH = './hg38.ml.fa'

LOCI_PATH = './sequences.bed'
BIGWIG_PATH = './bigwig_folder'
ANNOT_FILE_PATH =  './experiments.tab'
TARGET_LENGTH = 896

TRAIN_CHROMOSOMES = [*range(1, 24, 2), 'X'] # train on odd chromosomes
VALID_CHROMOSOMES = [*range(2, 24, 2)]      # validate on even

HELD_OUT_TARGET = ['SOX2']

# instantiate enformer or load pretrained

enformer = Enformer.from_pretrained('EleutherAI/enformer-official-rough', target_length = TARGET_LENGTH)

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
    fasta_file = FASTA_FILE_PATH,
    train_chromosome_ids = TRAIN_CHROMOSOMES,
    valid_chromosome_ids = VALID_CHROMOSOMES,
    held_out_targets = HELD_OUT_TARGET
)

# do gradient steps in a while loop

while True:
    _ = trainer()
```

## Resources

If you are low on GPU memory, you can save by making sure the protein and contextual embeddings are executed on CPU

```bash
CONTEXT_EMBED_USE_CPU=1 PROTEIN_EMBED_USE_CPU=1 python train.py
```

## Data

Transcription factor dataset

```python
from tf_bind_transformer.data import FactorProteinDataset

ds = FactorProteinDataset(
    folder = 'path/to/tfactor/fastas'
)

# single factor

ds['ETV1'] # <seq>

# multi-complexes

ds['PAX3-FOXO1'] # (<seq1>, <seq2>)

```

## Preprocessing (wip)

get a copy of hg38 blacklist bed file from calico

```bash
$ gsutil cp gs://basenji_barnyard/hg38.blacklist.rep.bed ./
```

using bedtools to filter out repetitive regions of the genome

```bash
$ bedtools intersect -v -a ./remap2022_all_macs2_hg38_v1_0.bed -b ./hg38.blacklist.rep.bed > remap2022_all_filtered.bed
```

## Caching

During training, protein sequences and contextual strings are cached to `~/.cache.tf.bind.transformer` directory. If you would like to make sure the caching is working, you just need to run your training script with `VERBOSE=1`

ex.

```bash
$ VERBOSE=1 python train.py
```

You can also force a cache clearance

```bash
$ CLEAR_CACHE=1 python train.py
```

## Todo

- [x] ESM and AF2 embedding fetching integrations
- [x] HF transformers integration for conditioning on free text
- [x] allow for fine-tuning layernorms of Enformer easily
- [x] add caching for external embeddings
- [x] figure out a way for external models (ESM, transformers) to be omitted from state dictionary on saving (use singletons)
- [x] take care of caching genetic sequences when enformer is frozen
- [x] offer a fully transformer variant with cross-attention with shared attention matrix and FiLM conditioning with contextual embed
- [x] also offer using pooled genetic / protein sequence concatted with context -> project -> squeeze excitation type conditioning
- [x] use checkpointing when fine-tuning enformer
- [x] take care of prepping dataframe with proper chromosome and training / validation split
- [x] use basenji blacklist bed file for filtering out rows in remap
- [x] filter remap dataframe based on tfactor fasta folder
- [x] filter remap dataframe with hg38 blacklist
- [x] handle targets with modifications from remap with all peaks (underscore in name)
- [x] grad clipping
- [x] add a safe initialization whereby rows of dataframe with targets not found in the tfactor fasta folder will be filtered out
- [x] add accuracy metric to fine tune script
- [x] master trainer class that handles both training / validation splitting, efficient instantiation of dataframe, filtering etc
- [x] write a simple trainer class that takes care of the training loop
- [x] create faster protein and context embedding derivation by optionally moving model to gpu and back to cpu when done
- [x] use ProtTrans for longer context proteins, look into AF2
- [x] make protalbert usable with one flag
- [x] log auxiliary losses appropriately (read value)
- [x] write fine-tuning script for finetuning on merged genomic track(s) from remap
- [ ] support for custom transformers other than enformer
- [ ] warmup in training loop
- [ ] mixed precision
- [ ] use wandb for experiment tracking
- [ ] cleanup tech debt in data and protein_utils
- [ ] explore protein model fine-tuning of layernorm
- [ ] auto-auroc calc
- [ ] k-fold cross validation
- [ ] output attention intermediates (or convolution output for hypertransformer), for interpreting binding site
- [ ] use prefect.io to manage downloading of tfactors fastas, remap scoped negative peaks, blacklist filtering etc

## Appreciation

This work was generously sponsored by <a href="https://github.com/jeffhsu3">Jeff Hsu</a> to be done completely open sourced.

## Citations

```bibtex
@article {Avsec2021.04.07.438649,
    author  = {Avsec, {\v Z}iga and Agarwal, Vikram and Visentin, Daniel and Ledsam, Joseph R. and Grabska-Barwinska, Agnieszka and Taylor, Kyle R. and Assael, Yannis and Jumper, John and Kohli, Pushmeet and Kelley, David R.},
    title   = {Effective gene expression prediction from sequence by integrating long-range interactions},
    elocation-id = {2021.04.07.438649},
    year    = {2021},
    doi     = {10.1101/2021.04.07.438649},
    publisher = {Cold Spring Harbor Laboratory},
    URL     = {https://www.biorxiv.org/content/early/2021/04/08/2021.04.07.438649},
    eprint  = {https://www.biorxiv.org/content/early/2021/04/08/2021.04.07.438649.full.pdf},
    journal = {bioRxiv}
}
```

```bibtex
@misc{yao2021filip,
    title   = {FILIP: Fine-grained Interactive Language-Image Pre-Training},
    author  = {Lewei Yao and Runhui Huang and Lu Hou and Guansong Lu and Minzhe Niu and Hang Xu and Xiaodan Liang and Zhenguo Li and Xin Jiang and Chunjing Xu},
    year    = {2021},
    eprint  = {2111.07783},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{tay2020hypergrid,
    title   = {HyperGrid: Efficient Multi-Task Transformers with Grid-wise Decomposable Hyper Projections},
    author  = {Yi Tay and Zhe Zhao and Dara Bahri and Donald Metzler and Da-Cheng Juan},
    year    = {2020},
    eprint  = {2007.05891},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```

```bibtex
@misc{lowe2021logavgexp,
    title   = {LogAvgExp Provides a Principled and Performant Global Pooling Operator},
    author  = {Scott C. Lowe and Thomas Trappenberg and Sageev Oore},
    year    = {2021},
    eprint  = {2111.01742},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@article{10.1093/nar/gkab996,
    author  = {Hammal, Fayrouz and de Langen, Pierre and Bergon, Aurélie and Lopez, Fabrice and Ballester, Benoit},
    title   = "{ReMap 2022: a database of Human, Mouse, Drosophila and Arabidopsis regulatory regions from an integrative analysis of DNA-binding sequencing experiments}",
    journal = {Nucleic Acids Research},
    issn    = {0305-1048},
    doi     = {10.1093/nar/gkab996},
    url     = {https://doi.org/10.1093/nar/gkab996},
    eprint  = {https://academic.oup.com/nar/article-pdf/50/D1/D316/42058627/gkab996.pdf},
}
```
