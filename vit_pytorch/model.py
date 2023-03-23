# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_model.ipynb.

# %% auto 0
__all__ = ['VisionTransformer']

# %% ../nbs/02_model.ipynb 3
import torch
from torch import nn
import torch.functional as F
from torchvision import datasets
import numpy as np

import yaml
from fastcore.basics import Path

# %% ../nbs/02_model.ipynb 16
from .patch import PatchEmbedding
from .encoder import TransformerEncoder
import torchvision.transforms as T

# %% ../nbs/02_model.ipynb 18
class VisionTransformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        n_classes = config["model"]["n_classes"]
        training = config["model"]["training"]
        emb_dim = config["patch"]["out_ch"]
        dropout = config["model"]["clf_dropout"]
        hidden_units = config["model"]["clf_hidden_units"]
        self.patch_embedding = PatchEmbedding(config)
        self.transformer_encoder = TransformerEncoder(config)
        # classification head
        self.ln = nn.LayerNorm(emb_dim)
        mlp_layers = (
            [
                nn.Linear(emb_dim, hidden_units),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_units, n_classes),
            ]
            if training
            else [nn.Linear(emb_dim, n_classes)]
        )
        self.mlp = nn.Sequential(self.ln, *mlp_layers)
        self.representation_ = None
        self.class_token_ = None

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        self.representation_ = x[:, 1:, :] # learned representation
        # ? In lucidrains implementation, why is class token same in vision transformer and repeated in bs
        # ? In my implementation, I initialized classtoken for each image, pass only the class token through the mlp head (bs, 1, 768)
        # this is the first item that was concatenated in the patch embedding
        self.class_token_ = self.patch_embedding.class_token
        print(self.class_token_.shape)
        x = self.mlp(self.class_token_)
        return x
