## Transcription Factor binding predictions with Attention and Transformers

A repository with exploration into using transformers to predict DNA ↔ transcription factor binding. If you are a researcher interested in this problem, please join the <a href="https://discord.gg/s7WyNU24aM">discord server</a> for discussions.

## Install

```bash
$ pip install tf-bind-transformer
```

## Usage

```python
import torch
from tf_bind_transformer import Model

# instantiate enformer or load pretrained

from enformer_pytorch import Enformer
enformer = Enformer(
    dim = 1536,
    depth = 2,
    target_length = 256
)

# instantiate model wrapper that takes in enformer

model = Model(
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

print(corr_coef.shape)
```

## Todo

- [ ] ESM and AF2 embedding fetching integrations
- [ ] allow for fine-tuning layernorms of Enformer easily
- [ ] normalization of interactions between genetic and amino acid sequence
- [ ] hyperparameters for different types of normalization on fine grained interactions feature map
- [ ] offer different ways of conditioning both paths with contextual embedding (hypergrid, ISAB cross attention, etc)
- [ ] HF transformers integration for conditioning on free text

## Appreciation

This work is generously sponsored by <a href="https://github.com/jeffhsu3">Jeff Hsu</a> to be done completely open sourced.

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
