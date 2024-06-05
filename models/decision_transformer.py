import torch 
import torch.nn as nn
from transformers import DecisionTransformerConfig, DecisionTransformerModel
from ..config import cfg
import numpy as np


class DecisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = DecisionTransformerConfig(state_dim=512,
                                       act_dim=5,
                                       hidden_size=1152,
                                       max_ep_len=cfg.train.hparams.max_ep_len)

        self.patch_pos_embeds = nn.Linear(4, 512, bias=False)
        
        self.model = DecisionTransformerModel(self.config)

        self.criterion = nn.CrossEntropyLoss()
    
    def get_coord_feats(self, patch_sequence, patch_size=5):
        y = patch_sequence//patch_size
        x = patch_sequence%patch_size
        rel_pos = torch.stack((torch.sin(2*np.pi*x/(patch_size-1)), torch.sin(2*np.pi*y/(patch_size-1)), torch.cos(2*np.pi*x/(patch_size-1)), torch.cos(2*np.pi*y/(patch_size-1))), dim=2)
        return rel_pos
    
    def forward(self, states, actions, rewards, returns_to_go, timesteps, patch_sequence, compute_loss=False):
        rel_pos = self.patch_pos_embeds(self.get_coord_feats(patch_sequence))
        states = states+rel_pos
        state_preds, action_preds, return_preds = self.model(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        return_dict=False,
        )

        if compute_loss:
            loss = self.criterion(action_preds.squeeze(0), actions.squeeze(0))
            return action_preds, loss
        return action_preds
