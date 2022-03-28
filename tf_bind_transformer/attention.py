import torch
from torch import nn
from einops import rearrange
from torch import einsum
from bidirectional_cross_attention import BidirectionalCrossAttention

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

# self attention

class SelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        mask = None,
    ):
        h = self.heads
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q = q * self.scale

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(mask):
            mask_value = -torch.finfo(sim.dtype).max
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dropout = 0.,
        ff_mult = 4,
        **kwargs
    ):
        super().__init__()
        self.attn = SelfAttention(dim = dim, dropout = dropout, **kwargs)
        self.ff = FeedForward(dim = dim, mult = ff_mult, dropout = dropout)

    def forward(self, x, mask = None):
        x = self.attn(x, mask = mask) + x
        x = self.ff(x) + x
        return x

# directional cross attention

class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        context_dim = None,
        dropout = 0.
    ):
        super().__init__()
        context_dim = default(context_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        context,
        mask = None,
        context_mask = None
    ):
        h = self.heads

        x = self.norm(x)
        context = self.context_norm(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q = q * self.scale

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(context_mask):
            mask_value = -torch.finfo(sim.dtype).max
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~context_mask, mask_value)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class JointCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        context_dim = None,
        ff_mult = 4,
        dropout = 0.,
        **kwargs
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.attn = BidirectionalCrossAttention(dim = dim, context_dim = context_dim, dropout = dropout, prenorm = True, **kwargs)
        self.ff = FeedForward(dim, mult = ff_mult, dropout = dropout)
        self.context_ff = FeedForward(context_dim, mult = ff_mult, dropout = dropout)

    def forward(
        self,
        x,
        context,
        mask = None,
        context_mask = None
    ):
        attn_out, context_attn_out = self.attn(x, context, mask = mask, context_mask = context_mask)

        x = x + attn_out
        context = context + context_attn_out

        x = self.ff(x) + x
        context = self.context_ff(context) + context

        return x, context
