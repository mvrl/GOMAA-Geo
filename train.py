import os

import torch
import numpy as np
from .models import PPO
from .data_utils import Sequence

from .utils import seed_everything, get_dist, generate_random_dist_config
from .config import cfg, action_list

device = torch.device('cuda:0')

import torch
import torch.nn.functional as F
import numpy as np


if __name__=='__main__':

    if not os.path.exists(os.path.join(cfg.train.ckpt_folder, cfg.train.expt_folder)):
        os.makedirs(os.path.join(cfg.train.ckpt_folder, cfg.train.expt_folder))

    import json

    with open(os.path.join(cfg.train.ckpt_folder, cfg.train.expt_folder, "config.json"), "w") as f:
        json.dump(cfg, f)

    seed_everything(cfg.train.hparams.random_seed)

    print_freq = 0

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(cfg.train.hparams.lr_actor,
    cfg.train.hparams.lr_critic,
    cfg.train.hparams.lr_llm,
    cfg.train.hparams.gamma,
    cfg.train.hparams.K_epochs,
    cfg.train.hparams.eps_clip,
    cfg.train.hparams.lr_gamma).cuda()

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    flag = 1
    num_success = 0

    file = open(os.path.join(cfg.train.ckpt_folder, cfg.train.expt_folder, cfg.train.log_name), "a+")

    average_steps_to_success = []
    average_deviation_from_opt = []
    average_reward = []
    num_successess = []

    dataset_dict = np.load(cfg.data.train_path, allow_pickle=True)

    val_success=0

    config = generate_random_dist_config(cfg.data.val_path, dist_possible=[7,8])

    if cfg.train.load_from_checkpoint:
        ppo_agent.load_state_dict(torch.load(cfg.train.checkpoint_path))
    

    while time_step <= cfg.train.hparams.max_training_timesteps:

        current_ep_reward = 0

        for i in range(len(dataset_dict[()].keys())):

            seq = Sequence(dataset_dict[()][f"img_{i}"], num_patches=cfg.data.patch_size)

            dist = np.random.randint(1, 9)
            GOAL_PATCH = np.random.randint(0, cfg.data.patch_size**2)
            CURRENT_PATCH = np.random.randint(0, cfg.data.patch_size**2)
            while get_dist(CURRENT_PATCH, GOAL_PATCH) != dist:#GOAL_PATCH == CURRENT_PATCH:
                GOAL_PATCH = np.random.randint(0, cfg.data.patch_size**2)
                CURRENT_PATCH = np.random.randint(0, cfg.data.patch_size**2)


            optimal_steps = get_dist(CURRENT_PATCH, GOAL_PATCH)
            best_dist = optimal_steps
            seq.init_with_goal_image(GOAL_PATCH)
            seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)

            for t in (range(1, np.random.randint(optimal_steps, cfg.train.hparams.max_ep_len+1))):
                inputs = seq.get_input_for_model(device='cuda:0') 
                if inputs["actions"] == []:
                    state = ppo_agent.llm(inputs_embeds=inputs["inputs_embeds"],
                    patch_sequence=inputs["patch_sequence"][:, 1:],
                    patch_size=cfg.data.patch_size)
                else:
                    state = ppo_agent.llm(
                        inputs_embeds=inputs["inputs_embeds"],
                        actions=[inputs["actions"]],
                        patch_sequence=inputs["patch_sequence"][:, 1:],
                        patch_size=cfg.data.patch_size)

                # select action with policy
                action = ppo_agent.select_action(state, seq.patch_sequence, cfg.data.patch_size)
                seq.update_sequence_with_action(action_list[action])
                
                current_patch_id = seq.patch_sequence[-1]
                prev_patch_id = seq.patch_sequence[-2]
                goal_patch_id = seq.patch_sequence[0]

                reward = ppo_agent.get_reward(cfg.data.patch_size, prev_patch_id, current_patch_id, goal_patch_id, seq.patch_sequence[1:-1], action, best_dist)
                if reward==1:
                    best_dist = get_dist(current_patch_id, goal_patch_id)
                done = (current_patch_id==GOAL_PATCH)
                if done:
                    average_steps_to_success.append(len(seq.action_sequence))
                    average_deviation_from_opt.append(len(seq.action_sequence)-optimal_steps)
                    num_success+=1
                
                # saving reward and is_terminals
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                
                time_step +=1
                current_ep_reward += reward

                # update PPO agent
                if time_step % cfg.train.hparams.update_timestep == 0:
                    ppo_agent.update(True, seq.patch_sequence, cfg.data.patch_size)
                
                # break; if the episode is over
                if done:
                    break 

        print_freq+=1

        num_successess.append(num_success)
        num_success = 0

        ppo_agent.eval()
        cur_val_success = ppo_agent.validate(config, cfg.data.val_path)
        if cur_val_success >= val_success:
            torch.save(ppo_agent.state_dict(), os.path.join(cfg.train.ckpt_folder, cfg.train.expt_folder, cfg.train.expt_name))
            val_success = cur_val_success
        ppo_agent.train()

        if print_freq%2 == 0:
            # print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)
            average_reward.append(print_avg_reward)

            if i_episode<20:
                print("Episode : {} \t Timestep : {} \t Average Reward : {} \t Num Successess : {} \t Success Ratio : {} \t Average Steps : {} \t Deviation OPT : {}".format(i_episode, time_step, np.mean(average_reward[-20:])/len(dataset_dict[()].keys()), sum(num_successess[-20:]), sum(num_successess[-20:])/((i_episode+1)*len(dataset_dict[()].keys())),np.mean(average_steps_to_success[-200:]), np.mean(average_deviation_from_opt[-200:])))
            else:
                print("Episode : {} \t Timestep : {} \t Average Reward : {} \t Num Successess : {} \t Success Ratio : {} \t Average Steps : {} \t Deviation OPT : {}".format(i_episode, time_step, np.mean(average_reward[-20:])/len(dataset_dict[()].keys()), sum(num_successess[-20:]), sum(num_successess[-20:])/(20*len(dataset_dict[()].keys())),np.mean(average_steps_to_success[-200:]), np.mean(average_deviation_from_opt[-200:])))

            print_running_reward = 0
            print_running_episodes = 0

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        i_episode += 1

        file.write(f"CURRENT_PATCH: {CURRENT_PATCH}, GOAL_PATCH: {GOAL_PATCH}\n")
        file.write(str(seq.patch_sequence))
        file.write("\n")
        file.write(str(seq.action_sequence))
        file.write("\n\n")