import copy
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from functools import wraps

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from contextlib import contextmanager

from enformer_pytorch import Enformer
from enformer_pytorch.modeling_enformer import poisson_loss, pearson_corr_coef
from enformer_pytorch.finetune import freeze_batchnorms_, freeze_all_but_layernorms_, unfreeze_last_n_layers_, unfreeze_all_layers_

from logavgexp_pytorch import logavgexp

from tf_bind_transformer.cache_utils import cache_fn
from tf_bind_transformer.protein_utils import get_protein_embedder
from tf_bind_transformer.context_utils import get_text_repr, get_contextual_dim

from tf_bind_transformer.attention import FeedForward, JointCrossAttentionBlock, CrossAttention, SelfAttentionBlock

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def identity(fn, *args, **kwargs):
    return fn

@contextmanager
def null_context():
    yield

# tensor helpers

def l2norm(t):
    return F.normalize(t, dim = -1)

def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def fourier_encode(x, dims, theta = 20000):
    device, dtype = x.device, x.dtype
    emb = math.log(theta) / (dims // 2)
    emb = torch.exp(torch.arange(dims // 2, device = device) * -emb)
    emb = rearrange(x, 'n -> n 1') * rearrange(emb, 'd -> 1 d')
    emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
    return emb

def corr_coef_loss(pred, target):
    return 1 - pearson_corr_coef(pred, target).mean()

# genetic sequence caching enformer forward decorator

def cache_enformer_forward(fn):
    cached_forward = cache_fn(fn, clear = True, path = 'genetic')

    @wraps(fn)
    def inner(seqs, *args, **kwargs):
        if seqs.ndim == 3:
            seqs = seqs.argmax(dim = -1)

        seq_list = seqs.unbind(dim = 0)
        seq_cache_keys = [''.join(list(map(str, one_seq.tolist()))) for one_seq in seq_list]
        outputs = [cached_forward(one_seq, *args, __cache_key = seq_cache_key, **kwargs) for one_seq, seq_cache_key in zip(seq_list, seq_cache_keys)]
        return torch.stack(outputs)

    return inner

# model

class FiLM(nn.Module):
    def __init__(
        self,
        dim,
        conditioned_dim
    ):
        super().__init__()
        self.to_gamma = nn.Linear(dim, conditioned_dim)
        self.to_bias = nn.Linear(dim, conditioned_dim)

    def forward(self, x, condition, mask = None):
        gamma = self.to_gamma(condition)
        bias = self.to_bias(condition)

        x = x * rearrange(gamma, 'b d -> b 1 d')
        x = x + rearrange(bias, 'b d -> b 1 d')
        return x

class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        dim,
        conditioned_dim,
        eps = 1e-8
    ):
        super().__init__()
        self.eps = eps
        self.to_gate = nn.Linear(dim + conditioned_dim, conditioned_dim)

    def forward(self, x, condition, mask = None):
        if exists(mask):
            numer = x.masked_fill(mask[..., None], 0.).sum(dim = 1)
            denom = mask.sum(dim = 1)[..., None].clamp(min = self.eps)
            mean_x = numer / denom
        else:
            mean_x = x.mean(dim = 1)

        condition = torch.cat((condition, mean_x), dim = -1)
        gate = self.to_gate(condition)

        x = x * rearrange(gate, 'b d -> b 1 d').sigmoid()
        return x

# read value MLP for calculating auxiliary loss

class ReadValueMLP(nn.Module):
    def __init__(
        self,
        dim,
        *,
        fourier_dims = 256,
        norm_factor_fourier = 50,
        norm_factor_linear = 8000,
        eps = 1e-20
    ):
        super().__init__()
        self.eps = eps
        self.fourier_dims = fourier_dims
        self.norm_factor_fourier = norm_factor_fourier
        self.norm_factor_linear = norm_factor_linear

        self.logits_norm = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(dim)
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim + fourier_dims + 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, 1),
            Rearrange('... 1 -> ...')
        )

    def forward(self, logits, peaks_nr, read_value):
        logits = self.logits_norm(logits)

        peaks_nr_log_space = torch.log(peaks_nr + self.eps)

        peaks_nr = rearrange(peaks_nr, '... -> (...)')
        peaks_nr_encoded = fourier_encode(peaks_nr / self.norm_factor_fourier, self.fourier_dims)
        peaks_nr_normed = rearrange(peaks_nr, '... -> ... 1') / self.norm_factor_linear

        peaks_nr_encoded_with_self = torch.cat((peaks_nr_normed, peaks_nr_log_space, peaks_nr_encoded), dim = -1)

        logits_with_peaks = torch.cat((logits, peaks_nr_encoded_with_self), dim = -1)

        pred = self.mlp(logits_with_peaks)
        read_value = rearrange(read_value, '... -> (...)')

        return F.smooth_l1_loss(pred, read_value)

class HypergridLinear(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        context_dim
    ):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(dim, dim_out))
        self.contextual_projection = nn.Linear(context_dim, dim * dim_out)

    def forward(self, x, context):
        # derive contextual gating, from hypergrids paper

        gating = self.contextual_projection(context).sigmoid()
        gating = rearrange(gating, 'b (i o) -> b i o', i = int(math.sqrt(gating.shape[-1])))

        # gate interactions projection with context

        to_logits_w = rearrange(self.weights, 'i o -> 1 i o') * gating
        return einsum('b n d, b d e -> b n e', x, to_logits_w)

# FILIP adapter model

class FILIP(nn.Module):
    def __init__(
        self,
        dim,
        context_dim,
        heads,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        inner_latent_dim = heads * dim_head

        self.to_latent_w = nn.Parameter(torch.randn(dim, inner_latent_dim))
        self.to_latent_b = nn.Parameter(torch.randn(inner_latent_dim))

        self.pre_attn_dropout = dropout

        self.null_context = nn.Parameter(torch.randn(heads, dim_head))
        self.context_to_latent_w = nn.Parameter(torch.randn(context_dim, inner_latent_dim))
        self.context_to_latent_b = nn.Parameter(torch.randn(inner_latent_dim))

    def forward(
        self,
        x,
        context,
        context_mask = None
    ):
        b, heads, device = x.shape[0], self.heads, x.device

        x = einsum('b n d, d e -> b n e', x, self.to_latent_w)
        x = x + self.to_latent_b

        x = rearrange(x, 'b n (h d) -> b h n d', h = heads)

        context = einsum('b n d, d e -> b n e', context, self.context_to_latent_w)
        context = context + self.context_to_latent_b

        context = rearrange(context, 'b n (h d) -> b h n d', h = heads)

        context, x = map(l2norm, (context, x))

        # fine grained interaction between dna and protein sequences
        # FILIP https://arxiv.org/abs/2111.07783

        if x.shape[0] == 1:
            # in the case one passes in 1 genomic sequence track
            # but multiple factors + contexts, as in enformer training
            x = rearrange(x, '1 ... -> ...')
            einsum_eq = 'h i d, b h j d -> b h i j'
        else:
            einsum_eq = 'b h i d, b h j d -> b h i j'

        # create context mask if not exist

        if not exists(context_mask):
            context_mask = torch.ones((b, context.shape[-1]), device = device).bool()

        # dropout mask by dropout prob

        if self.training:
            keep_mask = prob_mask_like(context_mask, 1 - self.pre_attn_dropout)
            context_mask = context_mask & keep_mask

        # add null context and modify mask

        context_mask = F.pad(context_mask, (1, 0), value = True)
        context_mask = rearrange(context_mask, 'b j -> b 1 1 j')

        null_context = repeat(self.null_context, 'h d -> b h 1 d', b = b)
        context = torch.cat((null_context, context), dim = -2)

        # differentiable max, as in FILIP paper

        interactions = einsum(einsum_eq, x, context)
        interactions = logavgexp(interactions, mask = context_mask, dim = -1, temp = 0.05)
        interactions = rearrange(interactions, 'b h i -> b i h')
        return interactions

class AdapterModel(nn.Module):
    def __init__(
        self,
        *,
        enformer,
        latent_dim = 64,
        latent_heads = 32,
        aa_embed_dim = None,
        aa_embed_encoder = 'esm',
        contextual_embed_dim = None,
        use_aa_embeds = False,
        use_free_text_context = False,
        free_text_context_encoder = 'pubmed',
        free_text_embed_method = 'cls',
        dropout = 0.,
        binary_target = False,
        target_mse_loss = False,
        aux_read_value_loss = False,
        read_value_aux_loss_weight = 0.05,
        joint_cross_attn_depth = 1,
        genome_self_attn_depth = 0,
        fourier_dims = 256,
        condition_squeeze_excite = False,
        condition_film = False,
        condition_hypergrid = True,
        use_corr_coef_loss = False,
        finetune_output_heads = None,
        **kwargs
    ):
        super().__init__()
        assert isinstance(enformer, Enformer), 'enformer must be an instance of Enformer'
        self.enformer = enformer
        enformer_dim = enformer.dim * 2

        if exists(finetune_output_heads):
            self.enformer.add_heads(**finetune_output_heads)

        self.norm_seq_embed = nn.LayerNorm(enformer_dim)

        # contextual embedding related variables

        assert free_text_embed_method in {'cls', 'mean_pool'}, 'must be either cls or mean_pool'
        self.free_text_embed_method = free_text_embed_method
        self.use_free_text_context = use_free_text_context

        if use_free_text_context:
            contextual_embed_dim = get_contextual_dim(free_text_context_encoder)
        else:
            assert exists(contextual_embed_dim), 'contextual embedding dimension must be given if not using transformer encoder'

        # protein embedding related variables

        self.use_aa_embeds = use_aa_embeds
        self.aa_embed_config = get_protein_embedder(aa_embed_encoder)
        self.get_aa_embed = self.aa_embed_config['fn']

        if use_aa_embeds:
            aa_embed_dim = self.aa_embed_config['dim']
        else:
            assert exists(aa_embed_dim), 'AA embedding dimensions must be set if not using ESM'

        # conditioning

        self.cond_genetic = None
        self.cond_protein = None

        if condition_squeeze_excite or condition_film:
            condition_klass = SqueezeExcitation if condition_squeeze_excite else FiLM

            self.cond_genetic  = condition_klass(contextual_embed_dim, enformer_dim)
            self.cond_protein  = condition_klass(contextual_embed_dim, aa_embed_dim)

        # genome self attn

        self.genome_self_attns = nn.ModuleList([])

        for _ in range(genome_self_attn_depth):
            attn = SelfAttentionBlock(
                dim = enformer_dim,
                dropout = dropout
            )
            self.genome_self_attns.append(attn)

        # joint attn

        self.joint_cross_attns = nn.ModuleList([])

        for _ in range(joint_cross_attn_depth):
            attn = JointCrossAttentionBlock(
                dim = enformer_dim,
                context_dim = aa_embed_dim,
                dropout = dropout
            )

            self.joint_cross_attns.append(attn)

        # latents

        self.filip = FILIP(
            dim = enformer_dim,
            context_dim = aa_embed_dim,
            dim_head = latent_dim,
            heads = latent_heads,
            dropout = dropout
        )

        # hypergrid conditioning

        if condition_hypergrid:
            self.linear_with_hypergrid = HypergridLinear(latent_heads, latent_heads, context_dim = contextual_embed_dim)
        else:
            self.linear_to_logits = nn.Linear(latent_heads, latent_heads)

        # to prediction

        self.binary_target = binary_target
        self.aux_read_value_loss = aux_read_value_loss
        self.read_value_aux_loss_weight = read_value_aux_loss_weight

        if binary_target:
            self.loss_fn = F.binary_cross_entropy_with_logits if not target_mse_loss else F.mse_loss

            self.to_pred = nn.Sequential(
                Reduce('... n d -> ... d', 'mean'),
                nn.LayerNorm(latent_heads),
                nn.Linear(latent_heads, 1),
                Rearrange('... 1 -> ...')
            )

            self.to_read_value_aux_loss = ReadValueMLP(
                dim = latent_heads,
                fourier_dims = fourier_dims
            )

        else:
            self.loss_fn = poisson_loss if not use_corr_coef_loss else corr_coef_loss

            self.to_pred = nn.Sequential(
                nn.Linear(latent_heads, 1),
                Rearrange('... 1 -> ...'),
                nn.Softplus()
            )

    def combine_losses(self, loss, aux_loss):
        if not self.aux_read_value_loss:
            return loss

        return loss + self.read_value_aux_loss_weight * aux_loss

    def forward_enformer_head(
        self,
        seq_embed,
        *,
        head,
        target = None,
        return_corr_coef = False
    ):
        assert not self.binary_target, 'cannot finetune on tracks if binary_target training is turned on'

        unfreeze_all_layers_(self.enformer._heads)

        assert head in self.enformer._heads, f'{head} head not found in enformer'

        pred = self.enformer._heads[head](seq_embed)

        if not exists(target):
            return pred

        assert pred.shape[-1] == target.shape[-1], f'{head} head on enformer produced {pred.shape[-1]} tracks, but the supplied target only has {target.shape[-1]}'

        if exists(target) and return_corr_coef:
            return pearson_corr_coef(pred, target)

        return self.loss_fn(pred, target)

    def forward(
        self,
        seq,
        *,
        aa = None,
        aa_embed = None,
        contextual_embed = None,
        contextual_free_text = None,
        aa_mask = None,
        target = None,
        read_value = None,
        peaks_nr = None,
        return_corr_coef = False,
        finetune_enformer = False,
        finetune_enformer_ln_only = False,
        unfreeze_enformer_last_n_layers = 0,
        head = None
    ):
        device = seq.device

        # prepare enformer for training
        # - set to eval and no_grad if not fine-tuning
        # - always freeze the batchnorms

        freeze_batchnorms_(self.enformer)
        enformer_forward = self.enformer.forward

        if finetune_enformer:
            enformer_context = null_context()
        elif finetune_enformer_ln_only:
            enformer_context = null_context()
            freeze_all_but_layernorms_(self.enformer)
        else:
            self.enformer.eval()
            enformer_context = torch.no_grad()
            enformer_forward_wrapper = cache_enformer_forward if self.training else identity
            enformer_forward = enformer_forward_wrapper(enformer_forward)

        # if unfreezing last N layers of enformer

        if unfreeze_enformer_last_n_layers > 0:
            unfreeze_last_n_layers_(self.enformer, unfreeze_enformer_last_n_layers)

        # genetic sequence embedding

        with enformer_context:
            seq_embed = enformer_forward(seq, return_only_embeddings = True)

        # if training off an enformer head

        if exists(head):
            return self.forward_enformer_head(seq_embed, head = head, target = target)

        # norm sequence embedding

        seq_embed = self.norm_seq_embed(seq_embed)

        for self_attn_block in self.genome_self_attns:
            seq_embed = self_attn_block(seq_embed)

        # protein related embeddings

        if self.use_aa_embeds:
            assert exists(aa), 'aa must be passed in as tensor of integers from 0 - 20 (20 being padding)'
            aa_embed, aa_mask = self.get_aa_embed(aa, device = seq.device)
        else:
            assert exists(aa_embed), 'protein embeddings must be given as aa_embed'

        # free text embeddings, for cell types and experimental params

        if not exists(contextual_embed):
            assert self.use_free_text_context, 'use_free_text_context must be set to True if one is not passing in contextual_embed tensor'
            assert exists(contextual_free_text), 'context must be supplied as array of strings as contextual_free_text if contextual_embed is not supplied'

            contextual_embed = get_text_repr(
                contextual_free_text,
                return_cls_token = (self.free_text_embed_method == 'cls'),
                device = seq.device
            )

        # contextual conditioning
        # film or squeeze-excite for both genetic and protein sequences

        if exists(self.cond_genetic):
            seq_embed = self.cond_genetic(seq_embed, contextual_embed)

        if exists(self.cond_protein):
            aa_embed = self.cond_protein(aa_embed, contextual_embed, mask = aa_mask)

        # joint cross attention

        for cross_attn in self.joint_cross_attns:
            seq_embed, aa_embed = cross_attn(
                seq_embed,
                context = aa_embed,
                context_mask = aa_mask
            )

        # project both embeddings into shared latent space

        interactions = self.filip(
            seq_embed,
            aa_embed,
            context_mask = aa_mask
        )


        # linear with hypergrid conditioning

        if exists(self.linear_with_hypergrid):
            logits = self.linear_with_hypergrid(interactions, context = contextual_embed)
        else:
            logits = self.linear_to_logits(interactions)

        # to *-seq prediction

        pred = self.to_pred(logits)

        if not exists(target):
            return pred

        if exists(target) and return_corr_coef:
            return pearson_corr_coef(pred, target)

        if exists(target) and not self.binary_target:
            return self.loss_fn(pred, target)

        # binary loss w/ optional auxiliary loss

        loss = self.loss_fn(pred, target.float())

        if not self.aux_read_value_loss:
            return loss, torch.Tensor([0.]).to(device)

        # return prediction if not auto-calculating loss

        assert exists(read_value) and exists(peaks_nr), 'peaks NR must be supplied if doing auxiliary read value loss'

        aux_loss = self.to_read_value_aux_loss(
            logits,
            peaks_nr,
            read_value = read_value
        )

        return loss, aux_loss
