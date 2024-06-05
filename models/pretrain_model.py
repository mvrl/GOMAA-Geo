#from .model import get_model
#from .model_mamba import get_model
#from .model_llama2 import get_model
#from .model_gemma import get_model
#from .model_mixtral import get_model
#from .model_gpt import get_model
from .model_falcon import get_model
from ..config import cfg
import torch 
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def neg_log(x):
    return -torch.log(x + 1e-5)

class MaskedActionModeling(LightningModule):
    def __init__(self, train_dataset, val_dataset, **kwargs):
        super().__init__()
        self.llm, *_ = get_model(cfg.train.llm_model, cfg.train.num_actions, cfg.train.llm_hidden_dim)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = kwargs.get('batch_size', 1)
        self.num_workers = kwargs.get('num_workers', 16)
        self.lr = kwargs.get('lr', 1e-5)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, embeds, action_seq, patch_seq, gt_action):
        state = self.llm(
                        inputs_embeds=embeds,
                        actions=action_seq,
                        patch_sequence=patch_seq[:, 1:],
                        patch_size=cfg.data.patch_size,
                        pretrain=True)

        loss_action = self.criterion(state, gt_action.float())
        
        return loss_action

    def shared_step(self, batch, batch_idx):
        embeds, action_seq, patch_seq, gt_action = batch
        loss  = self(embeds, np.array(action_seq).T.tolist(), patch_seq, gt_action)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        return {"loss": loss}

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                        shuffle=True,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        persistent_workers=True,
                        pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                        shuffle=False,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        persistent_workers=True,
                        pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=cfg.pretrain.hparams.lr, weight_decay=cfg.pretrain.hparams.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, cfg.pretrain.hparams.warmup)
        return [optimizer], [scheduler]