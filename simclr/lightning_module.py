import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision
import numpy as np
import pytorch_lightning as pl
import tqdm
import os
import timm.optim.optim_factory as optim_factory
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from functools import partial
from ..common.pos_embeddings import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed
from ..common.kwt import Transformer


layernorm_wrapper = partial(nn.LayerNorm, eps=1e-6)

# A thin wrapper over KWT
# Add's a projection head `g(.)` to the output of the KWT
class KWTWrapper(nn.Module):
    def __init__(self, 
                 input_res, 
                 patch_res,
                 dim=128,
                 # model_embed_dim=256,
                 g_dim=128,
                 encoder_depth=4,
                 encoder_heads=4,
                 encoder_mlp_dim=64,
                 pool='cls',
                 channels=1,
                 dropout=0., 
                 pre_norm=True, 
                 mode='ssl',
                 **kwargs):
        super().__init__()
        num_patches = int(input_res[0] / patch_res[0] * input_res[1] / patch_res[1])
        patch_dim = channels * patch_res[0] * patch_res[1]
        self.patchify = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_res[0], p2=patch_res[1])
        self.to_patch_embedding = nn.Sequential(
            self.patchify,
            nn.Linear(patch_dim, dim),
        )
        self.input_res = input_res
        self.patch_res = patch_res

        total_patches = self.get_num_patches()
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        total_patches += 1
        self.total_patches = total_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, total_patches, dim), requires_grad=False)
        self.encoder = Transformer(dim, encoder_depth, encoder_heads, encoder_mlp_dim, pre_norm, dropout)
        self.encoder_norm = layernorm_wrapper(dim)
        self.embed_dim = dim
        

        self.mode = mode
        if self.mode == "ssl":
            self.g = nn.Sequential(
                nn.Linear(dim, g_dim*4, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(g_dim*4, g_dim),
            )
        else:
            self.g = nn.Identity()

    def grid_size(self):
        return int(self.input_res[0] / self.patch_res[0]), int(self.input_res[1] / self.patch_res[1])

    def get_num_patches(self):
        grid = self.grid_size()
        return grid[0]*grid[1]
    
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size(), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x, hidden_states, attentions = self.encoder(x)
        x = self.encoder_norm(x)

        # take the cls token
        x = x[:,0,:]

        x = self.g(x)
        return x


class SimCLR_KWT(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.mode = kwargs['mode']
        self.kwt = KWTWrapper(**kwargs)
        
        if self.mode != "ssl":
            self.classifier = nn.Linear(self.kwt.embed_dim, kwargs['num_classes'])
        self.lr = kwargs['lr']
        self.wd = kwargs['weight_decay']
        self.epochs = kwargs['num_epochs']
        self.crop_transforms = torchvision.transforms.RandomResizedCrop(kwargs['input_res'], scale=(0.2, 1.0), 
                                                                        interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT)
        self.audio_transforms = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(20, iid_masks=True),
            torchaudio.transforms.TimeMasking(20, iid_masks=True),
        )
        self.temperature = kwargs.get("temperature", 0.1)

    def info_nce_loss(self, x, mode='train'):
        # create two augmented views of the input
        x1 = self.audio_transforms(self.crop_transforms(x))
        x2 = self.audio_transforms(self.crop_transforms(x))
        # append them into a single batch
        batch = torch.cat([x1,x2], dim=0)
        # Encode all "images"
        feats = self.kwt(batch)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        self.log(mode + "_loss", nll, prog_bar=True, on_epoch=True)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())
        return nll

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        if self.mode == "ssl":
            loss = self.info_nce_loss(x, mode='train')
            # already logged
        else:
            preds = self.classifier(self.kwt(x))
            loss = nn.functional.cross_entropy(preds, y)
            acc = (preds.argmax(dim=-1) == y).float().mean()
            self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        if self.mode == "ssl":
            loss = self.info_nce_loss(x, mode='val')
            # already logged
        else:
            preds = self.classifier(self.kwt(x))
            loss = nn.functional.cross_entropy(preds, y)
            acc = (preds.argmax(dim=-1) == y).float().mean()
            self.log("val_acc", acc, prog_bar=True, on_epoch=True)
            self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.mode == "ssl":
          loss = self.info_nce_loss(x, mode='test')
          # already logged
        else:
            preds = self.classifier(self.kwt(x))
            loss = nn.functional.cross_entropy(preds, y)
            acc = (preds.argmax(dim=-1) == y).float().mean()
            self.log("test_acc", acc)
            self.log('test_loss', loss)

    def configure_optimizers(self):
        if self.mode == "ssl":
            param_groups = optim_factory.param_groups_weight_decay(self, self.wd)
            optimizer = torch.optim.AdamW(param_groups, lr=self.lr, weight_decay=self.wd,
                              betas=(0.9, 0.95))
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs, eta_min=self.lr / 50
            )
            return [optimizer], [lr_scheduler]
        elif self.mode == "supervised" or self.mode == 'finetune':
            return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            raise ValueError(f"Unsupported value {self.mode} provided for mode. Should be one of ['ssl','supervised','finetune']")


