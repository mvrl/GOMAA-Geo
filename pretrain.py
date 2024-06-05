from .models import MaskedActionModeling
from .utils import PretrainRandomSequences
from .config import cfg
import torch 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import os


if __name__=='__main__':

    torch.cuda.empty_cache()
    train_dataset = PretrainRandomSequences(np.load(cfg.data.train_path, allow_pickle=True), cfg.data.patch_size, cfg.pretrain.min_seq_length)
    val_dataset = PretrainRandomSequences(np.load(cfg.data.val_path, allow_pickle=True), cfg.data.patch_size, cfg.pretrain.min_seq_length)

    checkpoint = ModelCheckpoint(
            monitor='val_loss',
            dirpath=os.path.join(cfg.pretrain.ckpt_folder, cfg.pretrain.expt_folder),
            filename=cfg.pretrain.expt_name,
            mode='min'
        )

    model = MaskedActionModeling(train_dataset, val_dataset)

    trainer = pl.Trainer(
        accelerator=cfg.pretrain.hparams.accelerator,
        devices=cfg.pretrain.hparams.devices,
        max_epochs=cfg.pretrain.hparams.epochs,
        num_nodes=1,
        callbacks=[checkpoint],
        accumulate_grad_batches=256,
        log_every_n_steps=256
        )
    trainer.fit(model)