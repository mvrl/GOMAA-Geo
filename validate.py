import os

import torch
from .models import PPO
from .data_utils import Sequence

from .utils import generate_config, seed_everything
from .config import cfg

device = torch.device('cuda:0')

if __name__=='__main__':

    if not os.path.exists(os.path.join(cfg.train.ckpt_folder, cfg.train.expt_folder)):
        os.makedirs(os.path.join(cfg.train.ckpt_folder, cfg.train.expt_folder))

    import json

    with open(os.path.join(cfg.train.ckpt_folder, cfg.train.expt_folder, "config.json"), "w") as f:
        json.dump(cfg, f)

    seed_everything(cfg.train.hparams.random_seed)

    # initialize a PPO agent
    ppo_agent = PPO(cfg.train.hparams.lr_actor,
    cfg.train.hparams.lr_critic,
    cfg.train.hparams.lr_llm,
    cfg.train.hparams.gamma,
    cfg.train.hparams.K_epochs,
    cfg.train.hparams.eps_clip,
    cfg.train.hparams.lr_gamma).cuda()

    ppo_agent.load_state_dict(torch.load(cfg.train.checkpoint_path))
    
    ppo_agent.eval()

    valid_path = cfg.data.test_path
    ground_path = cfg.data.ground_embeds_path
    ground_text_path = cfg.data.text_embeds_path

    import time

    start = time.time()

    for d in range(4, 9):
        seed_everything(cfg.train.hparams.random_seed)
        config = generate_config(cfg.data.test_path, patch_size=cfg.data.patch_size, dist=d, n_config_per_img=5)
    
        cur_val_success, res = ppo_agent.validate(config, valid_path, n_config_per_img=5)

        ## Goal Mask ##

        #cur_val_success = ppo_agent.validate_mask_goal(config, valid_path)

        ## Random ##

        #cur_val_success = ppo_agent.validate_random(config, valid_path, n_config_per_img=5)
        
        ## Multi Modal: ground##

        #cur_val_success = ppo_agent.validate_ground_level_custom(config, valid_path, ground_path, n_config_per_img=5)
        
        ## Multi Modal: text##

        #cur_val_success = ppo_agent.validate_ground_level_custom(config, valid_path, ground_text_path, n_config_per_img=5)

        ## Varying budget ##

        #cur_val_success = ppo_agent.validate_varying_budget(config, valid_path, n_config_per_img=5)
        
        print(f"dist={d}", f"success_ratio: {cur_val_success/890}") #xbd: 800
        print(res)
        print(time.time()-start)