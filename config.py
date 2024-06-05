from easydict import EasyDict as edict

action_list = {0:"up",
1:"right",
2:"down",
3:"left"
}

cfg = edict()

cfg.data = edict()
cfg.data.patch_size = 5
cfg.data.train_path = 'gomaa_geo/data/papr_train_sat_embeds_grid_5.npy'
cfg.data.val_path = 'gomaa_geo/data/papr_val_sat_embeds_grid_5.npy'
cfg.data.test_path = 'gomaa_geo/data/papr_val_sat_embeds_grid_5.npy'
cfg.data.ground_embeds_path = 'gomaa_geo/data/papr_my_ground_embeds.npy'
cfg.data.text_embeds_path = 'gomaa_geo/data/papr_my_text_embeds.npy'

cfg.pretrain = edict()
cfg.pretrain.ckpt_folder = "gomaa_geo/checkpoint"
cfg.pretrain.expt_folder = "pretrain_falcon"
cfg.pretrain.expt_name ="sat2cap_optimal_action_falcon.pt"
cfg.pretrain.log_name = "expt_logs.txt"
cfg.pretrain.min_seq_length = 6

cfg.pretrain.hparams = edict()
cfg.pretrain.hparams.accelerator='gpu'
cfg.pretrain.hparams.lr = 1e-5
cfg.pretrain.hparams.warmup = 5
cfg.pretrain.hparams.devices = 1
cfg.pretrain.hparams.epochs = 3000
cfg.pretrain.hparams.weight_decay = 0.0001


cfg.train = edict()
cfg.train.llm_checkpoint = "gomaa_geo/checkpoint/pretrain_falcon/sat2cap_optimal_action_falcon.pt.ckpt"
cfg.train.load_from_checkpoint = False
cfg.train.checkpoint_path = "gomaa_geo/checkpoint/sat2cap_falcon/ppo_falcon.pt"
cfg.train.ckpt_folder = "gomaa_geo/checkpoint"
cfg.train.expt_folder = "sat2cap_falcon"
cfg.train.expt_name ="ppo_falcon.pt"
cfg.train.log_name = "expt_logs.txt"
cfg.train.llm_model = "tiiuae/falcon-7b"
cfg.train.num_actions = 4
cfg.train.llm_hidden_dim = 1152

cfg.train.hparams = edict()
cfg.train.hparams.max_ep_len = 10
cfg.train.hparams.max_training_timesteps = int(1e8)
cfg.train.hparams.log_freq = cfg.train.hparams.max_ep_len * 2
cfg.train.hparams.save_model_freq = int(2e4)
cfg.train.hparams.update_timestep = cfg.train.hparams.max_ep_len * 64
cfg.train.hparams.K_epochs = 4
cfg.train.hparams.eps_clip = 0.2
cfg.train.hparams.gamma = 0.93
cfg.train.hparams.lr_actor = 0.0001
cfg.train.hparams.lr_critic = 0.0001
cfg.train.hparams.lr_llm = 0.0001
cfg.train.hparams.lr_gamma = 0.9999
cfg.train.hparams.random_seed = 42
