import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm.optim.optim_factory as optim_factory
from .mae import MAE, mae_loss
from ..common.schedulers import LinearWarmupCosineAnnealingLR


class MAELightning(pl.LightningModule):
  def __init__(self, **kwargs):
    super().__init__()
    self.save_hyperparameters()
    self.mae = MAE(**kwargs)
    self.mode = self.mae.mode
    if self.mode == "supervised" or self.mode == "finetune":
      self.classifier = nn.Linear(
          self.mae.embed_dim, kwargs['num_classes']
      )
    self.lr = kwargs['lr']
    self.wd = kwargs['weight_decay']
    self.epochs = kwargs['num_epochs']

    if self.mode == "ssl":
        num_tr_samples = kwargs['num_samples']
        self.num_tr_iters = int(num_tr_samples / kwargs['batch_size'])
        self.warmup_epochs = kwargs['warmup_epochs']

  def forward(self, x):
    if self.mode == "ssl":
      return self.mae(x)
    else:
      x = self.mae(x)
      x = self.classifier(x)
      return x

  def configure_optimizers(self):
    if self.mode == "ssl":

      # return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)
      param_groups = optim_factory.param_groups_weight_decay(self.mae, self.wd)
      optimizer = torch.optim.AdamW(param_groups, lr=self.lr, weight_decay=self.wd,
                              betas=(0.9, 0.95))
      scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                warmup_epochs=self.warmup_epochs*self.num_tr_iters,
                                                max_epochs=self.epochs*self.num_tr_iters)

      return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    elif self.mode == "supervised" or self.mode == 'finetune':
      return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
    else:
        raise ValueError(f"Unsupported value {self.mode} provided for mode. Should be one of ['ssl','supervised','finetune']")

  def training_step(self, train_batch, batch_idx):
    x, y = train_batch
    preds = self.forward(x)
    if self.mode == "ssl":
      pred, target, mask = preds
      loss = mae_loss(pred, target, mask)
    else:
      loss = nn.functional.cross_entropy(preds, y)
      acc = (preds.argmax(dim=-1) == y).float().mean()
      self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
    return loss

  def validation_step(self, val_batch, batch_idx):
    x, y = val_batch
    preds = self.forward(x)
    if self.mode == "ssl":
      pred, target, mask = preds
      loss = mae_loss(pred, target, mask)
    else:
      loss = nn.functional.cross_entropy(preds, y)
      acc = (preds.argmax(dim=-1) == y).float().mean()
      self.log("val_acc", acc, prog_bar=True, on_epoch=True)
    self.log('val_loss', loss, prog_bar=True, on_epoch=True)

  def test_step(self, batch, batch_idx):
    x, y = batch
    preds = self.forward(x)
    if self.mode == "ssl":
      pred, target, mask = preds
      loss = mae_loss(pred, target, mask)
    else:
      loss = nn.functional.cross_entropy(preds, y)
      acc = (preds.argmax(dim=-1) == y).float().mean()
      self.log("test_acc", acc)
    self.log('test_loss', loss)
