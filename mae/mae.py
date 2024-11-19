import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from functools import partial
from ..common.pos_embeddings import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed
from ..common.kwt import Transformer

layernorm_wrapper = partial(nn.LayerNorm, eps=1e-6)


class MAE(nn.Module):
  def __init__(self, 
               input_res, 
               patch_res,
               num_classes=None,  # placeholder,  not used
               dim=128,
               encoder_depth=4,
               encoder_heads=4,
               encoder_mlp_dim=64,
               decoder_depth=1,
               decoder_embed_dim=128,
               decoder_heads=4,
               decoder_mlp_dim=64,
               pool='cls',
               mode='ssl',
               masking_mode='unstructured',
               channels=1,
               dropout=0., 
               pre_norm=True, 
               masking_ratio=0.8, 
               **kwargs):
    super().__init__()
    num_patches = int(input_res[0] / patch_res[0] * input_res[1] / patch_res[1])

    patch_dim = channels * patch_res[0] * patch_res[1]
    assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

    self.patchify = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_res[0], p2=patch_res[1])

    self.to_patch_embedding = nn.Sequential(
        self.patchify,
        nn.Linear(patch_dim, dim),
    )
    self.input_res = input_res
    self.patch_res = patch_res

    total_patches = self.get_num_patches()
    print("Total patches:", total_patches)

    self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
    total_patches += 1

    self.total_patches = total_patches
    self.pos_embed = nn.Parameter(torch.zeros(1, total_patches, dim), requires_grad=False)

    self.mask_ratio = masking_ratio

    self.encoder = Transformer(dim, encoder_depth, encoder_heads, encoder_mlp_dim, pre_norm, dropout)
    self.encoder_norm = layernorm_wrapper(dim)

    self.embed_dim = dim

    if mode == "ssl":
      self.decoder_embed = nn.Linear(dim, decoder_embed_dim, bias=True)
      self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
      self.decoder_pos_embed = nn.Parameter(torch.zeros(1, total_patches, decoder_embed_dim), requires_grad=False)

      self.decoder = Transformer(decoder_embed_dim, decoder_depth, decoder_heads, decoder_mlp_dim, pre_norm, dropout)

      self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim, bias=True)
      self.decoder_norm = layernorm_wrapper(decoder_embed_dim)
    self.mode = mode
    self.masking_mode = masking_mode
    self.initialize_weights()

  def grid_size(self):
    return int(self.input_res[0] / self.patch_res[0]), int(self.input_res[1] / self.patch_res[1])

  def get_num_patches(self):
    grid = self.grid_size()
    return grid[0]*grid[1]

  def initialize_weights(self):
    pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size(), cls_token=True)
    self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    torch.nn.init.normal_(self.cls_token, std=0.02)
    if self.mode == "ssl":
        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], (self.total_patches-1), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=0.02)
    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

  def random_masking(self, x, mask_ratio):
      N, L, D = x.shape
      len_keep = int(L * (100 - mask_ratio*100) / 100)

      if self.masking_mode == "unstructured":
        noise = torch.rand(N, L, device=x.device)
      elif self.masking_mode == "timestep":
        grid_size = self.grid_size()
        noise = torch.rand(N, L//grid_size[1], device=x.device)
        noise = torch.repeat_interleave(noise, repeats=grid_size[1], dim=1)
      else:
        raise NotImplementedError(f"masking_mode={self.masking_mode} is not implemented")

      ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
      ids_restore = torch.argsort(ids_shuffle, dim=1)

      # keep the first subset
      ids_keep = ids_shuffle[:, :len_keep]
      x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

      # generate the binary mask: 0 is keep, 1 is remove
      mask = torch.ones([N, L], device=x.device)
      mask[:, :len_keep] = 0
      # unshuffle to get the binary mask
      mask = torch.gather(mask, dim=1, index=ids_restore)

      return x_masked, mask, ids_restore

  def forward_encoder(self, x, mask_ratio):
    x = self.to_patch_embedding(x)
    x = x + self.pos_embed[:, 1:, :]

    x, mask, ids_restore = self.random_masking(x, mask_ratio)

    cls_token = self.cls_token + self.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x, hidden_states, attentions = self.encoder(x)
    x = self.encoder_norm(x)
    return x, mask, ids_restore

  def forward_decoder(self, x, ids_restore):
    x = self.decoder_embed(x)
    mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

    x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

    x = x + self.decoder_pos_embed

    x, hidden_states, attentions = self.decoder(x)
    x = self.decoder_norm(x)
    x = self.decoder_pred(x)

    x = x[:, 1:, :]
    return x

  def pretrain(self, imgs):
    x, mask, ids_restore = self.forward_encoder(imgs, self.mask_ratio)
    pred = self.forward_decoder(x, ids_restore)
    target = self.patchify(imgs)
    return pred, target, mask

  def forward(self, x):
    if self.mode == "ssl":
      return self.pretrain(x)
    else:
      B = x.shape[0]
      x = self.to_patch_embedding(x)
      x = x + self.pos_embed[:, 1:, :]

      cls_token = self.cls_token + self.pos_embed[:, :1, :]
      cls_tokens = cls_token.expand(x.shape[0], -1, -1)
      x = torch.cat((cls_tokens, x), dim=1)
      x, _, _ = self.encoder(x)
      x = self.encoder_norm(x)

      outcome = x[:, 0, :]
      return outcome

def mae_loss(pred, target, mask, norm_pix_loss:bool=False):
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var+1e-6) ** 0.5
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)
    loss = (loss * mask).sum() / mask.sum()
    return loss
