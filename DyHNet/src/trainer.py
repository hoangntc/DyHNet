import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def build_trainer(config):
    if 'name_suffix' in config:
        name = '{}_{}'.format(config['name'], config['name_suffix'])
    else: 
        name = config['name']
    # callbacks
    checkpoint = ModelCheckpoint(
        dirpath=config['checkpoint_dir'], 
        filename=f'model={name}-' + '{epoch}-{val_loss:.4f}-{val_acc:.4f}-{val_hamming_loss:.4f}-{val_macro_f1:.4f}-{val_micro_f1:.4f}',
        save_top_k=config['top_k'],
        verbose=True,
        monitor=config['metric'],
        mode=config['mode'],
    )
    early_stopping = EarlyStopping(
        monitor=config['metric'], 
        min_delta=0.00, 
        patience=config['patience'],
        verbose=False,
        mode=config['mode'],
    )
    
    callbacks = [checkpoint, early_stopping]
    # trainer_kwargs
    trainer_kwargs = {
        'max_epochs': config['max_epochs'],
        'gpus': [0] if torch.cuda.is_available() else 0,
        'weights_summary': 'full',
        'callbacks': callbacks,
        'log_every_n_steps': 20,
    }

    trainer = Trainer(**trainer_kwargs)
    return trainer, trainer_kwargs