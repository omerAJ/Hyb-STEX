## definitions taken from i-jepa

import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn

import sys
sys.path.append('.')
sys.path.append('..')
from model.vision_transformer_utils import (
    trunc_normal_,
    repeat_interleave_batch,
    apply_masks_ctxt,
    apply_masks_targets,
    apply_masks_indices,
)

def get_2d_sincos_pos_embed(embed_dim, rows, cols, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(rows, dtype=float)
    grid_w = np.arange(cols, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, rows, cols])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features 
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, indices):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        qk = [q, k]
        attn1 = (q @ k.transpose(-2, -1)) * self.scale
        # import matplotlib.pyplot as plt
        # plt.imshow(indices[0, 0, :, :].cpu().numpy())
        # plt.colorbar(label='Attention Weight')
        # plt.show()
        if indices is not None:
            # print("NOT masking attention matrix")
            # print(f"indices.shape: {indices.shape}")  ## [1, 200, 1]
            nodes = indices.clone()
            nodes = nodes.expand_as(attn1)
            indices = indices.transpose(-1, -2)
            indices = indices.expand_as(attn1)#.transpose(-1, -2) ## expand to [b, 200, 200] by repeating rows
            # attn1[indices] = -torch.inf
            attn1[nodes & ~nodes.transpose(-1, -2)] = -torch.inf    ## bad rows and good cols
            attn1[~nodes & nodes.transpose(-1, -2)] = -torch.inf    ## good rows and bad cols
        
        attn = attn1.softmax(dim=-1)
        attn1 = [attn1, indices]
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn1, qk


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, indices, return_attention=False):
        y, attn, qk = self.attn(self.norm1(x), indices)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn, qk
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=[224, 224], patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ConvEmbed(nn.Module):
    """
    3x3 Convolution stems for ViT following ViTC models
    """

    def __init__(self, channels, strides, img_size=224, in_chans=3, batch_norm=True):
        super().__init__()
        # Build the stems
        stem = []
        channels = [in_chans] + channels
        for i in range(len(channels) - 2):
            stem += [nn.Conv2d(channels[i], channels[i+1], kernel_size=3,
                               stride=strides[i], padding=1, bias=(not batch_norm))]
            if batch_norm:
                stem += [nn.BatchNorm2d(channels[i+1])]
            stem += [nn.ReLU(inplace=True)]
        stem += [nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=strides[-1])]
        self.stem = nn.Sequential(*stem)

        # Comptute the number of patches
        stride_prod = int(np.prod(strides))
        self.num_patches = (img_size[0] // stride_prod)**2

    def forward(self, x):
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_seq_length, embedding_dim):
        """
        Initializes the learnable positional encoding module.
        
        Args:
        max_seq_length (int): The maximum length of the input sequences.
        embedding_dim (int): The dimensionality of the embeddings.
        """
        super(LearnablePositionalEncoding, self).__init__()
        # Define a learnable embedding layer
        self.position_embeddings = nn.Embedding(max_seq_length, embedding_dim)

    def forward(self, x):
        """
        Forward pass of the module.
        
        Args:
        x (torch.Tensor): Tensor of shape (batch_size, seq_length) containing the 
                          indices of the positions in the sequence.
        
        Returns:
        torch.Tensor: Tensor of shape (batch_size, seq_length, embedding_dim) containing
                      the positional embeddings.
        """
        # Get the positional embeddings
        position = torch.arange(x.size(1), device=x.device).unsqueeze(0).repeat(x.size(0), 1)
        pos_embeddings = self.position_embeddings(position)
        
        return pos_embeddings
    
class VisionTransformerPredictor(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=(20, 10),
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))    ## predictor_embed_dim -> embed_dim  Now adding pos embed to embed dim and then reducing to predictor_embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        self.num_patches = img_size[0] * img_size[1]

        # self.learnable_pos = LearnablePositionalEncoding(self.num_patches, predictor_embed_dim)

        self.pos_embed_learnable = nn.Parameter(torch.zeros(1, self.num_patches, predictor_embed_dim), requires_grad=True)
        nn.init.trunc_normal_(self.pos_embed_learnable, std=0.02)
        

        # self.predictor_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, predictor_embed_dim),
        #                                         requires_grad=False)
        # predictor_pos_embed = get_2d_sincos_pos_embed(self.predictor_pos_embed.shape[-1],
        #                                               img_size[0], img_size[1],
        #                                               cls_token=False)
        # self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_enc, masks_pred, pe):
        assert (masks_pred is not None) and (masks_enc is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_enc, list):
            # masks_enc = [masks_enc]
            pass

        if not isinstance(masks_pred, list):
            # masks_pred = [masks_pred]
            pass
        
        """
        indices = x<5
        indices = indices[0, 0, ...]
        indices = indices.view(-1, 1)  ## [200, 1]
        """

        # -- Batch Size
        # print("x.shape", x.shape, "masks_enc.shape", masks_enc.shape, "masks_pred.shape", masks_pred.shape)
        B = len(x) // masks_enc.shape[1]   ## masks_enc is [32, 1, 200], just in case  x: [32, 45, 256]

        # -- map from encoder-dim to pedictor-dim
        masked_pos_embed_learnable = apply_masks_ctxt(pe, masks_enc)
        # print(f"masked_learnable_pos_embed_x.shape: {masked_learnable_pos_embed_x.shape}")
        x += masked_pos_embed_learnable
        
        x = self.predictor_embed(x)    ## x: [32, 45, 256] -> [32, 45, pred_emb_dim]
        # print(f"x.shape: {x.shape}")
        # print(f"N: {N}")
        N = self.num_patches
        # temp = torch.arange(N).unsqueeze(0).repeat(B, 1).to(x.device)
        # print(f"temp.shape: {temp.shape}")
        # learnable_pos_embed_x = self.learnable_pos(temp)
        # -- add positional embedding to x tokens
        # x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)    
        # print(f"x.shape: {x.shape}", f"B: {B}", "x_pos_embed.shape", x_pos_embed.shape, "masks_enc.shape", masks_enc.shape)  
        # print(f"x.shape: {x.shape}", f"B: {B}", "learnable_pos_embed_x.shape", learnable_pos_embed_x.shape)
        # print(f"x.shape: {x.shape}")    ## [32, 30, 128]
        _, N_ctxt, D = x.shape   ## N_ctxt is the number of context tokens

        # -- concat mask tokens to x
        # pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        # temp = torch.arange(N).unsqueeze(0).repeat(B, 1).to(x.device)
        # learnable_pos_embed_target =  self.learnable_pos(temp)
        pos_embed_target_learnable =  pe
        
        # print("applying target mask on pos_embs", "pos_embs.shape", pos_embs.shape, "masks_pred.shape", masks_pred.shape) ## pos_embs:[32, 200, 128], masks_pred:[32, 4, 200]
        pos_embed_target_learnable = apply_masks_targets(pos_embed_target_learnable, masks_pred)
        # print("pos_embs.shape", pos_embs.shape)   ## [32*4, 6(number of tokens to select as target per sample per mask), 128]
        ## number of pred_tokens i need is 32*4*6 = 768
        
        # pos_embs = repeat_interleave_batch(pos_embs, B, repeat=masks_enc.shape[1])   ## repeat=no. of enc masks
        # print("pos_embs.shape", pos_embs.shape)   ## [32*4, 6(number of tokens to select as target per sample per mask), 128]
        # --
        pred_tokens = self.mask_token.repeat(pos_embed_target_learnable.size(0), pos_embed_target_learnable.size(1), 1)
        # print("pred_tokens.shape", pred_tokens.shape)   ## [32*4, 6(number of target tokens per sample per mask), 128]
        # --
        pred_tokens += pos_embed_target_learnable


        ## now send to predictor_embed_dim
        pred_tokens = self.predictor_embed(pred_tokens)

        # print(f"x.shape: {x.shape}")  ## [32, 42, 128]
        x = x.repeat(masks_pred.shape[1], 1, 1) 
        # print(f"x.shape (after repeat): {x.shape}")  
        x = torch.cat([x, pred_tokens], dim=1)   ## [32*4, 42+6, 128]
        # print(f"\n\nx.shape (after cat): {x.shape}")
        # indices = apply_masks_indices(indices, masks_pred)
        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x, indices=None)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        # print("x.shape", x.shape)  
        x = x[:, N_ctxt:]
        # print("x.shape", x.shape)
        x = self.predictor_proj(x)

        return x





class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        # --
        self.patch_embed = PatchEmbed(
            img_size=[img_size[0], img_size[1]],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        
        # self.up_projection = nn.Linear(2, embed_dim)
        num_patches = img_size[0] * img_size[1]

        # self.learnable_pos = LearnablePositionalEncoding(num_patches, embed_dim)
        self.pos_embed_learnable = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=True)
        nn.init.trunc_normal_(self.pos_embed_learnable, std=0.02)
        # --
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
        #                                     img_size[0], img_size[1],
        #                                     cls_token=False)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks=None, pe=None):
        if masks is not None:
            if not isinstance(masks, list):
                # masks = [masks]
                pass ## no need to convert mask to list

        ## get indices for nodes to mask in attention
        # print(f"x.shape: {x.shape}")
        # print(x)
        _, _, R, C = x.shape
        N = R*C
        _x = x.clone().sum(dim=1)
        # print(f"_x.shape: {_x.shape}")
        # print("\n_x: \n", _x)
        
        # indices = _x<-24.50  ## indices where bad sensor, dont keep these in attention
        # print(indices.shape)  ## [1, 70, 20, 10]
        # print(indices)
        # indices = indices[0, ...]
        # print("indices.shape: ", indices.shape)   ## [20, 10]
        # indices = indices.view(-1, 1)  ## [200, 1]
        # print("indices.shape: ", indices.shape)

        # -- patchify x
        x = self.patch_embed(x)
        upX = x.clone()
        # x = self.up_projection(x)
        # print("x.shape", x.shape)
        B, N, D = x.shape
        # x = torch.sum(x, dim=2)
        
        # -- add positional embedding to x
        # pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        
        
        """zero x"""
        # print(f"setting x=0")
        # x=torch.zeros_like(x)
        beforePosX = x.clone()
        
        # temp = torch.arange(N).unsqueeze(0).repeat(B, 1).to(x.device)
        # learnable_pos_embed = self.learnable_pos(temp)
        if pe is not None:
            pos_embed_learnable = pe
        else:
            pos_embed_learnable = self.pos_embed_learnable.repeat(B, 1, 1)
        x = x + pos_embed_learnable
        # x = x + pos_embed   ## add complete pos_emb before indexing
        
        posX = pos_embed_learnable.clone()
        # -- mask x
        if masks is not None:
            # print("\n\nbefore apply_masks_ctxt: \n\n", "x.shape", x.shape, "masks.shape", masks.shape) ## x: [32, 200, 256] masks: [32, 1, 200]
            # print(f"x.shape (before masks): {x.shape}")
            # print(f"indices.shape (before masks): {indices.shape}")
            x = apply_masks_ctxt(x, masks)
            # indices = apply_masks_indices(indices, masks)
            # print(f"x.shape (after masks): {x.shape}")
            # print(f"indices.shape (after masks): {indices.shape}")
            # print("\nafter apply_masks_ctxt: \n\n", "x.shape", x.shape)  ## [32, 45, 256]

        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x, attn, qk = blk(x, indices=None, return_attention=True)
            # x = blk(x)
            attn_list = [attn] if i == 0 else attn_list + [attn]
        if self.norm is not None:
            x = self.norm(x)

        return x, pos_embed_learnable, attn_list, upX, beforePosX, posX
        if pe is None:
            return x, pos_embed_learnable
        else:
            return x

    """check this"""
    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)
