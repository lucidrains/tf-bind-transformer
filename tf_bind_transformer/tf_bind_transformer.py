import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from enformer_pytorch import Enformer

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    return F.normalize(t, dim = -1)

# losses and metrics

def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()

def pearson_corr_coef(x, y, eps = 1e-8):
    x2 = x * x
    y2 = y * y
    xy = x * y
    ex = x.mean(dim = 1)
    ey = y.mean(dim = 1)
    exy = xy.mean(dim = 1)
    ex2 = x2.mean(dim = 1)
    ey2 = y2.mean(dim = 1)
    r = (exy - ex * ey) / (torch.sqrt(ex2 - (ex * ex)) * torch.sqrt(ey2 - (ey * ey)) + eps)
    return r.mean(dim = -1)

# model

class Model(nn.Module):
    def __init__(
        self,
        *,
        enformer,
        enformer_dim = 1536,
        latent_dim = 64,
        latent_heads = 32,
        aa_embed_dim = 512,
        contextual_embed_dim = 256,
    ):
        super().__init__()
        assert isinstance(enformer, Enformer), 'enformer must be an instance of Enformer'
        self.enformer = enformer

        self.latent_heads = latent_heads
        inner_latent_dim = latent_heads * latent_dim

        self.seq_embed_to_latent_w = nn.Parameter(torch.randn(enformer_dim * 2, inner_latent_dim))
        self.seq_embed_to_latent_b = nn.Parameter(torch.randn(inner_latent_dim))

        self.aa_seq_embed_to_latent_w = nn.Parameter(torch.randn(aa_embed_dim, inner_latent_dim))
        self.aa_seq_embed_to_latent_b = nn.Parameter(torch.randn(inner_latent_dim))

        self.contextual_projection = nn.Linear(contextual_embed_dim, latent_heads * latent_heads)

        self.to_pred = nn.Sequential(
            nn.Linear(latent_heads, 1),
            Rearrange('... 1 -> ...'),
            nn.Softplus()
        )

    def forward(
        self,
        seq,
        *,
        aa_embed,
        contextual_embed,
        aa_mask = None,
        target = None,
        return_corr_coef = False
    ):
        latent_heads = self.latent_heads
        _, seq_embed = self.enformer(seq, return_embeddings = True)

        # project both embeddings into shared latent space

        seq_latent = einsum('b n d, d e -> b n e', seq_embed, self.seq_embed_to_latent_w)
        seq_latent = seq_latent + self.seq_embed_to_latent_b

        seq_latent = rearrange(seq_latent, 'b n (h d) -> b h n d', h = latent_heads)

        aa_latent = einsum('b n d, d e -> b n e', aa_embed, self.aa_seq_embed_to_latent_w)
        aa_latent = aa_latent + self.aa_seq_embed_to_latent_b

        aa_latent = rearrange(aa_latent, 'b n (h d) -> b h n d', h = latent_heads)

        aa_latent, seq_latent = map(l2norm, (aa_latent, seq_latent))

        # fine grained interaction between dna and protein sequences
        # FILIP https://arxiv.org/abs/2111.07783

        interactions = einsum('b h i d, b h j d -> b h i j', seq_latent, aa_latent)

        # use mean pooling along amino acid sequence length

        if exists(aa_mask):
            aa_mask = rearrange(aa_mask, 'b j -> b 1 1 j')
            interactions = interactions.masked_fill(aa_mask, 0.)

        interactions = reduce(interactions, 'b h i j -> b i h', 'mean')

        # derive contextual projection

        hyper_weights = self.contextual_projection(contextual_embed)
        hyper_weights = rearrange(hyper_weights, 'b (i o) -> b i o', i = int(math.sqrt(hyper_weights.shape[-1])))

        # project interactions with hyper weights

        logits = einsum('b n d, b d e -> b n e', interactions, hyper_weights)
        pred = self.to_pred(logits)

        if exists(target):
            if return_corr_coef:
                return pearson_corr_coef(pred, target)

            return poisson_loss(pred, target)

        return pred
