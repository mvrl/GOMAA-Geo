import torch
import torch.nn as nn
from torch.distributions import Categorical
from ..data_utils import Sequence
from ..config import cfg, action_list
from .pretrain_model import MaskedActionModeling
from ..utils import get_dist
import numpy as np


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim=5):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
                        nn.Linear(state_dim, state_dim//4),
                        nn.Tanh(),
                        nn.Linear(state_dim//4, state_dim//4),
                        nn.Tanh(),
                        nn.Linear(state_dim//4, action_dim),
                        nn.Softmax(dim=-1)
                    )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, state_dim//4),
                        nn.Tanh(),
                        nn.Linear(state_dim//4, state_dim//4),
                        nn.Tanh(),
                        nn.Linear(state_dim//4, 1)
                    )

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, patch_sequence, patch_size):
        action_probs = self.actor(state)

        if patch_sequence[-1]%patch_size==0:
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 1.0, 1.0, 0.0]]).cuda())
        if patch_sequence[-1]%patch_size==patch_size-1:
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 0.0, 1.0, 1.0]]).cuda())
        if patch_sequence[-1] in torch.arange(0, patch_size):
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[0.0, 1.0, 1.0, 1.0]]).cuda())
        if patch_sequence[-1] in torch.arange(patch_size**2-patch_size, patch_size**2):
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 1.0, 0.0, 1.0]]).cuda())

        dist = Categorical(action_probs+1e-6)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action): 
        action_probs = self.actor(state)

        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

    def entropy_act(self, state, patch_sequence, patch_size):
        action_probs = self.actor(state)

        if patch_sequence[-1]%patch_size==0:
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 1.0, 1.0, 0.0]]).cuda())
        if patch_sequence[-1]%patch_size==patch_size-1:
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 0.0, 1.0, 1.0]]).cuda())
        if patch_sequence[-1] in torch.arange(0, patch_size):
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[0.0, 1.0, 1.0, 1.0]]).cuda())
        if patch_sequence[-1] in torch.arange(patch_size**2-patch_size, patch_size**2):
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 1.0, 0.0, 1.0]]).cuda())

        top_probs = torch.topk(action_probs, 2).values
        if (top_probs[0, 0]) > 0.6:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = Categorical(action_probs+1e-6)
            action = dist.sample()

        return action.item()
    
    def greedy_act(self, state, patch_sequence, patch_size):
        action_probs = self.actor(state)

        if patch_sequence[-1]%patch_size==0:
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 1.0, 1.0, 0.0]]).cuda())
        if patch_sequence[-1]%patch_size==patch_size-1:
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 0.0, 1.0, 1.0]]).cuda())
        if patch_sequence[-1] in torch.arange(0, patch_size):
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[0.0, 1.0, 1.0, 1.0]]).cuda())
        if patch_sequence[-1] in torch.arange(patch_size**2-patch_size, patch_size**2):
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 1.0, 0.0, 1.0]]).cuda())

        action = torch.argmax(action_probs, dim=-1)

        return action.item()
    
    def stochastic_act(self, state, patch_sequence, patch_size):
        action_probs = self.actor(state)

        if patch_sequence[-1]%patch_size==0:
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 1.0, 1.0, 0.0]]).cuda())
        if patch_sequence[-1]%patch_size==patch_size-1:
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 0.0, 1.0, 1.0]]).cuda())
        if patch_sequence[-1] in torch.arange(0, patch_size):
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[0.0, 1.0, 1.0, 1.0]]).cuda())
        if patch_sequence[-1] in torch.arange(patch_size**2-patch_size, patch_size**2):
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 1.0, 0.0, 1.0]]).cuda())

        dist = Categorical(action_probs+1e-6)
        action = dist.sample()

        return action.item()

    def random_act(self, state, patch_sequence, patch_size):
        action_probs = torch.FloatTensor([[0.25, 0.25, 0.25, 0.25]]).cuda()
        if patch_sequence[-1]%patch_size==0:
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 1.0, 1.0, 0.0]]).cuda())
        if patch_sequence[-1]%patch_size==patch_size-1:
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 0.0, 1.0, 1.0]]).cuda())
        if patch_sequence[-1] in torch.arange(0, patch_size):
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[0.0, 1.0, 1.0, 1.0]]).cuda())
        if patch_sequence[-1] in torch.arange(patch_size**2-patch_size, patch_size**2):
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 1.0, 0.0, 1.0]]).cuda())
        

        dist = Categorical(action_probs+1e-6)
        action = dist.sample()

        return action.item()
    

class PPO(nn.Module):
    def __init__(self, lr_actor, lr_critic, lr_llm, gamma, K_epochs, eps_clip, lr_gamma):
        super().__init__()
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.llm_module = MaskedActionModeling.load_from_checkpoint(
            cfg.train.llm_checkpoint,
            train_dataset=None,
            val_dataset=None,
        )

        self.llm = self.llm_module.llm

        state_dim = self.llm.config.word_embed_proj_dim
        action_dim = self.llm.config.num_actions

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.schedular = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, lr_gamma)

        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    ### required
    def select_action(self, state, patch_sequence, patch_size):
        
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(state, patch_sequence, patch_size)
            
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()
    
    def select_stochastic_action(self, state, patch_sequence, patch_size):
        return self.policy_old.stochastic_act(state, patch_sequence, patch_size)
    
    def select_greedy_action(self, state, patch_sequence, patch_size):
        return self.policy_old.greedy_act(state, patch_sequence, patch_size)
    
    def select_random_action(self, state, patch_sequence, patch_size):
        return self.policy_old.random_act(state, patch_sequence, patch_size)
    
    def select_entropy_action(self, state, patch_sequence, patch_size):
        return self.policy_old.entropy_act(state, patch_sequence, patch_size)
    

    def get_reward(self, patch_size, prev_patch_id, current_patch_id, goal_patch_id, patch_sequence, best_dist):

        cur_rows = current_patch_id//patch_size
        cur_cols = current_patch_id%patch_size

        goal_rows = goal_patch_id//patch_size
        goal_cols = goal_patch_id%patch_size


        if current_patch_id == prev_patch_id:
            return -1
        if cur_cols==goal_cols and cur_rows==goal_rows:
            return 2
        elif current_patch_id in patch_sequence:
            return -1
        elif (cur_rows - goal_rows)**2 + (cur_cols - goal_cols)**2 < best_dist:
            return 1
        else:
            return -1


    def update(self, flag, patch_sequence, patch_size, device="cuda:0"):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        self.schedular.step()
            
        # Copy new weights into old policy
        if True: #flag:
            self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def validate_varying_budget(self, config, valid_path, tokenizer=None, n_config_per_img=5, flag='none'):
        dataset_dict = np.load(valid_path, allow_pickle=True)
        total_imgs = len(dataset_dict[()].keys())
        num_success=0
        avg_dev_steps=0 
        avg_steps_to_success=0
        avg_reward=0
        res = [[0]*5 for _ in range(10)]

        for budget in range(5, 15):
            for i in range(total_imgs):
                for j in range(n_config_per_img):
                    seq = Sequence(dataset_dict[()][f"img_{i}"], tokenizer, num_patches=cfg.data.num_patches)
                    GOAL_PATCH = config[f"img_{i}"][j][0]
                    CURRENT_PATCH = config[f"img_{i}"][j][1]
                    seq.init_with_goal_image(GOAL_PATCH)
                    seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)
                    best_dist=get_dist(GOAL_PATCH, CURRENT_PATCH)
                    opt_steps=get_dist(GOAL_PATCH, CURRENT_PATCH)

                    for t in (range(1, budget+1)):

                        inputs = seq.get_input_for_model(device='cuda:0')

                        if inputs["actions"] == []:
                            state = self.llm(inputs_embeds=inputs["inputs_embeds"],
                            patch_sequence=inputs["patch_sequence"][:, 1:],
                            patch_size=cfg.data.patch_size)
                        else:
                            state = self.llm(
                                inputs_embeds=inputs["inputs_embeds"],
                                actions=[inputs["actions"]],
                                patch_sequence=inputs["patch_sequence"][:, 1:],
                                patch_size=cfg.data.patch_size)

                        if flag=='entropy':
                            action = self.select_entropy_action(state, seq.patch_sequence, cfg.data.patch_size)
                        else:
                            action = self.select_greedy_action(state, seq.patch_sequence, cfg.data.patch_size)

                        seq.update_sequence_with_action(action_list[action])
                        
                        current_patch_id = seq.patch_sequence[-1]
                        prev_patch_id = seq.patch_sequence[-2]
                        goal_patch_id = seq.patch_sequence[0]

                        reward = self.get_reward(cfg.data.patch_size, prev_patch_id, current_patch_id, goal_patch_id, seq.patch_sequence[1:-1], action, best_dist)
                        if reward==1:
                            best_dist=get_dist(current_patch_id, goal_patch_id)
                        avg_reward+=reward

                        done = (current_patch_id==GOAL_PATCH)
                        if done:
                            avg_steps_to_success+=len(seq.action_sequence)
                            avg_dev_steps+=(len(seq.action_sequence)-opt_steps)
                            num_success+=1
                            res[budget-5][j]+=1
                            break

        print(f"Val Success : {num_success} \t Val Avg Reward : {avg_reward/(total_imgs*5)} \t Val Avg Steps Success : {avg_steps_to_success/num_success} \t Val Dev : {avg_dev_steps/num_success}")
        return num_success, res


    def validate(self, config, valid_path, tokenizer=None, n_config_per_img=5, flag='none'):
        dataset_dict = np.load(valid_path, allow_pickle=True)
        total_imgs = len(dataset_dict[()].keys())
        num_success=0
        avg_dev_steps=0 
        avg_steps_to_success=0
        avg_reward=0
        res = [0]*5
        for i in range(total_imgs):
            for j in range(n_config_per_img):
                seq = Sequence(dataset_dict[()][f"img_{i}"], tokenizer, num_patches=5)
                GOAL_PATCH = config[f"img_{i}"][j][0]
                CURRENT_PATCH = config[f"img_{i}"][j][1]
                seq.init_with_goal_image(GOAL_PATCH)
                seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)
                best_dist=get_dist(GOAL_PATCH, CURRENT_PATCH)
                opt_steps=get_dist(GOAL_PATCH, CURRENT_PATCH)

                for t in (range(1, cfg.train.hparams.max_ep_len+1)):

                    inputs = seq.get_input_for_model(device='cuda:0')

                    if inputs["actions"] == []:
                        state = self.llm(inputs_embeds=inputs["inputs_embeds"],
                        patch_sequence=inputs["patch_sequence"][:, 1:],
                        patch_size=cfg.data.patch_size)
                    else:
                        state = self.llm(
                            inputs_embeds=inputs["inputs_embeds"],
                            actions=[inputs["actions"]],
                            patch_sequence=inputs["patch_sequence"][:, 1:],
                            patch_size=cfg.data.patch_size)

                    if flag=='entropy':
                        action = self.select_entropy_action(state, seq.patch_sequence, cfg.data.patch_size)
                    else:
                        action = self.select_greedy_action(state, seq.patch_sequence, cfg.data.patch_size)
                    seq.update_sequence_with_action(action_list[action])
                    
                    current_patch_id = seq.patch_sequence[-1]
                    prev_patch_id = seq.patch_sequence[-2]
                    goal_patch_id = seq.patch_sequence[0]

                    reward = self.get_reward(cfg.data.patch_size, prev_patch_id, current_patch_id, goal_patch_id, seq.patch_sequence[1:-1], action, best_dist)
                    if reward==1:
                        best_dist=get_dist(current_patch_id, goal_patch_id)
                    avg_reward+=reward
                    done = (current_patch_id==GOAL_PATCH)
                    if done:
                        avg_steps_to_success+=len(seq.action_sequence)
                        avg_dev_steps+=(len(seq.action_sequence)-opt_steps)
                        num_success+=1
                        res[j]+=1
                        break
        print(f"Val Success : {num_success} \t Val Avg Reward : {avg_reward/(total_imgs*5)} \t Val Avg Steps Success : {avg_steps_to_success/num_success} \t Val Dev : {avg_dev_steps/num_success}")
        return num_success, res

    def validate_ground_level_custom(self, config, sat_paths, ground_path, tokenizer=None, n_config_per_img=5):
        ground_dict = np.load(ground_path, allow_pickle=True)
        sat_dict = np.load(sat_paths, allow_pickle=True)
        total_imgs = len(sat_dict[()].keys())
        num_success=0
        avg_dev_steps=0 
        avg_steps_to_success=0
        avg_reward=0
        for i in range(total_imgs):
            for j in range(n_config_per_img):
                goal_ground = ground_dict[i]#ground_dict[()][f"img_{i}"]#ground_dict[i]#
                GOAL_PATCH = config[f"img_{i}"][j][0]
                CURRENT_PATCH = config[f"img_{i}"][j][1]
                x_min = 5 - GOAL_PATCH%5  
                x_max = x_min + 5
                y_min = 5 - GOAL_PATCH//5
                y_max = y_min + 5
                sat_embeds = sat_dict[()][f"img_{i}"][y_min:y_max, x_min:x_max].reshape(25, -1)
                seq = Sequence(sat_embeds, tokenizer, num_patches=5)

                seq.init_with_goal_embed(goal_ground, GOAL_PATCH)
                seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)
                best_dist=get_dist(GOAL_PATCH, CURRENT_PATCH)
                opt_steps=get_dist(GOAL_PATCH, CURRENT_PATCH)

                for t in (range(1, cfg.train.hparams.max_ep_len+1)):

                    inputs = seq.get_input_for_model(device='cuda:0')

                    if inputs["actions"] == []:
                        state = self.llm(inputs_embeds=inputs["inputs_embeds"],
                        patch_sequence=inputs["patch_sequence"][:, 1:],
                        patch_size=cfg.data.patch_size)
                    else:
                        state = self.llm(
                            inputs_embeds=inputs["inputs_embeds"],
                            actions=[inputs["actions"]],
                            patch_sequence=inputs["patch_sequence"][:, 1:],
                            patch_size=cfg.data.patch_size)

                    action = self.select_greedy_action(state, seq.patch_sequence, cfg.data.patch_size)
                    seq.update_sequence_with_action(action_list[action])
                    
                    current_patch_id = seq.patch_sequence[-1]
                    prev_patch_id = seq.patch_sequence[-2]
                    goal_patch_id = seq.patch_sequence[0]

                    reward = self.get_reward(cfg.data.patch_size, prev_patch_id, current_patch_id, goal_patch_id, seq.patch_sequence[1:-1], action, best_dist)
                    if reward==1:
                        best_dist=get_dist(current_patch_id, goal_patch_id)
                    avg_reward+=reward
                    done = (current_patch_id==GOAL_PATCH)
                    if done:
                        avg_steps_to_success+=len(seq.action_sequence)
                        avg_dev_steps+=(len(seq.action_sequence)-opt_steps)
                        num_success+=1
                        break
            print(seq.patch_sequence)
        print(f"Val Success : {num_success} \t Val Avg Reward : {avg_reward/(total_imgs*5)} \t Val Avg Steps Success : {avg_steps_to_success/num_success} \t Val Dev : {avg_dev_steps/num_success}")
        return num_success

    def validate_ground_level_custom_mask_goal(self, config, sat_paths, ground_path, tokenizer=None, n_config_per_img=5):
        ground_dict = np.load(ground_path, allow_pickle=True)
        sat_dict = np.load(sat_paths, allow_pickle=True)
        total_imgs = len(sat_dict[()].keys())
        num_success=0
        avg_dev_steps=0 
        avg_steps_to_success=0
        avg_reward=0
        for i in range(total_imgs):
            for j in range(n_config_per_img):
                goal_ground = ground_dict[()][f"img_{i}"]#ground_dict[i]
                GOAL_PATCH = config[f"img_{i}"][j][0]
                CURRENT_PATCH = config[f"img_{i}"][j][1]
                x_min = 5 - GOAL_PATCH%5  
                x_max = x_min + 5
                y_min = 5 - GOAL_PATCH//5
                y_max = y_min + 5
                sat_embeds = sat_dict[()][f"img_{i}"][y_min:y_max, x_min:x_max].reshape(25, -1)
                seq = Sequence(sat_embeds, tokenizer, num_patches=5)
                seq.init_with_goal_embed(goal_ground, GOAL_PATCH)
                seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)
                best_dist=get_dist(GOAL_PATCH, CURRENT_PATCH)
                opt_steps=get_dist(GOAL_PATCH, CURRENT_PATCH)

                for t in (range(1, cfg.train.hparams.max_ep_len+1)):

                    inputs = seq.get_input_for_model(device='cuda:0')
                    mask = torch.ones((1, len(inputs["inputs_embeds"][0])+len(inputs["actions"]))).to(inputs["inputs_embeds"].device)
                    mask[:, 0] = 0

                    if inputs["actions"] == []:
                        state = self.llm(inputs_embeds=inputs["inputs_embeds"],
                        patch_sequence=inputs["patch_sequence"][:, 1:],
                        attention_mask=mask,
                        patch_size=cfg.data.patch_size)
                    else:
                        state = self.llm(
                            inputs_embeds=inputs["inputs_embeds"],
                            actions=[inputs["actions"]],
                            patch_sequence=inputs["patch_sequence"][:, 1:],
                            attention_mask=mask,
                            patch_size=cfg.data.patch_size)

                    action = self.select_greedy_action(state, seq.patch_sequence, cfg.data.patch_size)
                    seq.update_sequence_with_action(action_list[action])
                    
                    current_patch_id = seq.patch_sequence[-1]
                    prev_patch_id = seq.patch_sequence[-2]
                    goal_patch_id = seq.patch_sequence[0]

                    reward = self.get_reward(cfg.data.patch_size, prev_patch_id, current_patch_id, goal_patch_id, seq.patch_sequence[1:-1], action, best_dist)
                    if reward==1:
                        best_dist=get_dist(current_patch_id, goal_patch_id)
                    avg_reward+=reward
                    done = (current_patch_id==GOAL_PATCH)
                    if done:
                        avg_steps_to_success+=len(seq.action_sequence)
                        avg_dev_steps+=(len(seq.action_sequence)-opt_steps)
                        num_success+=1
                        break
            print(seq.patch_sequence)
        print(f"Val Success : {num_success} \t Val Avg Reward : {avg_reward/(total_imgs*5)} \t Val Avg Steps Success : {avg_steps_to_success/num_success} \t Val Dev : {avg_dev_steps/num_success}")
        return num_success
    
    
    def validate_mask_goal(self, config, valid_path, tokenizer=None, n_config_per_img=5):
        dataset_dict = np.load(valid_path, allow_pickle=True)
        total_imgs = len(dataset_dict[()].keys())
        num_success=0
        avg_dev_steps=0 
        avg_steps_to_success=0
        avg_reward=0
        for i in range(total_imgs):
            for j in range(n_config_per_img):
                seq = Sequence(dataset_dict[()][f"img_{i}"], tokenizer, num_patches=5)
                GOAL_PATCH = config[f"img_{i}"][j][0]
                CURRENT_PATCH = config[f"img_{i}"][j][1]
                seq.init_with_goal_image(GOAL_PATCH)
                seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)
                best_dist=get_dist(GOAL_PATCH, CURRENT_PATCH)
                opt_steps=get_dist(GOAL_PATCH, CURRENT_PATCH)

                for t in (range(1, cfg.train.hparams.max_ep_len+1)):

                    inputs = seq.get_input_for_model(device='cuda:0')
                    mask = torch.ones((1, len(inputs["inputs_embeds"][0])+len(inputs["actions"]))).to(inputs["inputs_embeds"].device)
                    mask[:, 0] = 0

                    if inputs["actions"] == []:
                        state = self.llm(inputs_embeds=inputs["inputs_embeds"],
                        patch_sequence=inputs["patch_sequence"][:, 1:],
                        attention_mask=mask,
                        patch_size=cfg.data.patch_size)
                    else:
                        state = self.llm(
                            inputs_embeds=inputs["inputs_embeds"],
                            actions=[inputs["actions"]],
                            patch_sequence=inputs["patch_sequence"][:, 1:],
                            attention_mask=mask,
                            patch_size=cfg.data.patch_size)

                    action = self.select_greedy_action(state, seq.patch_sequence, cfg.data.patch_size)
                    seq.update_sequence_with_action(action_list[action])
                    
                    current_patch_id = seq.patch_sequence[-1]
                    prev_patch_id = seq.patch_sequence[-2]
                    goal_patch_id = seq.patch_sequence[0]

                    reward = self.get_reward(cfg.data.patch_size, prev_patch_id, current_patch_id, goal_patch_id, seq.patch_sequence[1:-1], action, best_dist)
                    if reward==1:
                        best_dist=get_dist(current_patch_id, goal_patch_id)
                    avg_reward+=reward
                    done = (current_patch_id==GOAL_PATCH)
                    if done:
                        avg_steps_to_success+=len(seq.action_sequence)
                        avg_dev_steps+=(len(seq.action_sequence)-opt_steps)
                        num_success+=1
                        break
        print(f"Val Success : {num_success} \t Val Avg Reward : {avg_reward/(total_imgs*5)} \t Val Avg Steps Success : {avg_steps_to_success/num_success} \t Val Dev : {avg_dev_steps/num_success}")
        return num_success
    
    def validate_random(self, config, tokenizer=None, n_config_per_img=5):
        dataset_dict = np.load(cfg.data.test_path, allow_pickle=True)
        total_imgs = len(dataset_dict[()].keys())
        opt_steps=4
        num_success=0
        avg_dev_steps=0 
        avg_steps_to_success=0
        avg_reward=0
        action_list = {0:"up",
                        1:"right",
                        2:"down",
                        3:"left"
                        }
        res = [0]*5
        for i in range(total_imgs):
            for j in range(n_config_per_img):
                seq = Sequence(dataset_dict[()][f"img_{i}"], tokenizer=None, num_patches=20)
                GOAL_PATCH = config[f"img_{i}"][j][0]
                CURRENT_PATCH = config[f"img_{i}"][j][1]
                seq.init_with_goal_image(GOAL_PATCH)
                seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)
                best_dist=get_dist(CURRENT_PATCH, GOAL_PATCH)

                for t in (range(1, cfg.train.hparams.max_ep_len+1)):

                    # select action with policy
                    ac = np.arange(0, 4)
                    if seq.patch_sequence[-1]%cfg.data.patch_size==0:
                        ac = np.delete(ac, np.where(ac == 3))
                    if seq.patch_sequence[-1]%cfg.data.patch_size==cfg.data.patch_size-1:
                        ac = np.delete(ac, np.where(ac == 1))
                    if seq.patch_sequence[-1] in torch.arange(0, cfg.data.patch_size):
                        ac = np.delete(ac, np.where(ac == 0))
                    if seq.patch_sequence[-1] in torch.arange(cfg.data.patch_size**2-cfg.data.patch_size, cfg.data.patch_size**2):
                        ac = np.delete(ac, np.where(ac == 2))

                    action = np.random.choice(ac)
                    seq.update_sequence_with_action(action_list[action])

                    inputs = seq.get_input_for_model(device='cuda:0')
                    
                    current_patch_id = seq.patch_sequence[-1]
                    prev_patch_id = seq.patch_sequence[-2]
                    goal_patch_id = seq.patch_sequence[0]

                    reward = self.get_reward(cfg.data.patch_size, prev_patch_id, current_patch_id, goal_patch_id, inputs["patch_sequence"][:, 1:-1], action, best_dist)
                    avg_reward+=reward
                    done = (current_patch_id==GOAL_PATCH)
                    if done:
                        avg_steps_to_success+=len(seq.action_sequence)
                        avg_dev_steps+=(len(seq.action_sequence)-best_dist)
                        num_success+=1
                        res[j]+=1
                        break
        print(f"Val Success : {num_success} \t Val Avg Reward : {avg_reward/(total_imgs*5)} \t Val Avg Steps Success : {avg_steps_to_success/num_success} \t Val Dev : {avg_dev_steps/num_success}")
        return num_success, res
    
    def validate_llm(self, config, tokenizer=None, n_config_per_img=5):
        dataset_dict = np.load(cfg.data.val_path, allow_pickle=True)
        total_imgs = len(dataset_dict[()].keys())
        
        num_success=0
        avg_dev_steps=0 
        avg_steps_to_success=0
        avg_reward=0
        for i in range(total_imgs):
            for j in range(n_config_per_img):
                seq = Sequence(dataset_dict[()][f"img_{i}"], tokenizer, num_patches=5)
                GOAL_PATCH = config[f"img_{i}"][j][0]
                CURRENT_PATCH = config[f"img_{i}"][j][1]
                seq.init_with_goal_image(GOAL_PATCH)
                seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)
                best_dist=get_dist(GOAL_PATCH, CURRENT_PATCH)
                opt_steps=get_dist(GOAL_PATCH, CURRENT_PATCH)

                for t in (range(1, cfg.train.hparams.max_ep_len+1)):

                    inputs = seq.get_input_for_model(device='cuda:0')

                    if inputs["actions"] == []:
                        state = self.llm(inputs_embeds=inputs["inputs_embeds"],
                        patch_sequence=inputs["patch_sequence"][:, 1:],
                        patch_size=cfg.data.patch_size)
                    else:
                        state = self.llm(
                            inputs_embeds=inputs["inputs_embeds"],
                            actions=[inputs["actions"]],
                            patch_sequence=inputs["patch_sequence"][:, 1:],
                            patch_size=cfg.data.patch_size)
                    
                    action = torch.argmax(self.llm.distance_pred(state).reshape(-1)).item()
                    seq.update_sequence_with_action(action_list[action])
                    
                    current_patch_id = seq.patch_sequence[-1]
                    prev_patch_id = seq.patch_sequence[-2]
                    goal_patch_id = seq.patch_sequence[0]

                    reward = self.get_reward(cfg.data.patch_size, prev_patch_id, current_patch_id, goal_patch_id, seq.patch_sequence[1:-1], action, best_dist)
                    if reward==1:
                        best_dist=get_dist(current_patch_id, goal_patch_id)
                    avg_reward+=reward
                    done = (current_patch_id==GOAL_PATCH)
                    if done:
                        avg_steps_to_success+=len(seq.action_sequence)
                        avg_dev_steps+=(len(seq.action_sequence)-opt_steps)
                        num_success+=1
                        break
        print(f"Val Success : {num_success} \t Val Avg Reward : {avg_reward/(total_imgs*5)} \t Val Avg Steps Success : {avg_steps_to_success/num_success} \t Val Dev : {avg_dev_steps/num_success}")
        return num_success
    

    def visualize_paths(self, valid_path, idx, start, goal, num_paths=100):
        dataset_dict = np.load(valid_path, allow_pickle=True)
        total_imgs = len(dataset_dict[()].keys())
        num_success=0
        avg_dev_steps=0 
        avg_steps_to_success=0
        avg_reward=0
        paths = {}
        optimal_path = []
        for j in range(num_paths+1):
            seq = Sequence(dataset_dict[()][f"img_{idx}"], num_patches=5)
            GOAL_PATCH = goal
            CURRENT_PATCH = start
            seq.init_with_goal_image(GOAL_PATCH)
            seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)
            best_dist=get_dist(GOAL_PATCH, CURRENT_PATCH)
            opt_steps=get_dist(GOAL_PATCH, CURRENT_PATCH)

            for t in (range(1, cfg.train.hparams.max_ep_len+1)):

                inputs = seq.get_input_for_model(device='cuda:0')

                if inputs["actions"] == []:
                    state = self.llm(inputs_embeds=inputs["inputs_embeds"],
                    patch_sequence=inputs["patch_sequence"][:, 1:],
                    patch_size=cfg.data.patch_size)
                else:
                    #mask = torch.ones((1, len(inputs["inputs_embeds"][0])+len(inputs["actions"]))).to(inputs["inputs_embeds"].device)
                    #mask[:, 1:-1] = 0
                    state = self.llm(
                        inputs_embeds=inputs["inputs_embeds"],
                        actions=[inputs["actions"]],
                        patch_sequence=inputs["patch_sequence"][:, 1:],
                        #attention_mask=mask,
                        patch_size=cfg.data.patch_size)
                
                if j==num_paths:
                    action = self.select_greedy_action(state, seq.patch_sequence, cfg.data.patch_size)
                else:
                    action = self.select_stochastic_action(state, seq.patch_sequence, cfg.data.patch_size)
                seq.update_sequence_with_action(action_list[action])
                
                current_patch_id = seq.patch_sequence[-1]
                prev_patch_id = seq.patch_sequence[-2]
                goal_patch_id = seq.patch_sequence[0]

                reward = self.get_reward(cfg.data.patch_size, prev_patch_id, current_patch_id, goal_patch_id, seq.patch_sequence[1:-1], action, best_dist)
                if reward==1:
                    best_dist=get_dist(current_patch_id, goal_patch_id)
                avg_reward+=reward
                done = (current_patch_id==GOAL_PATCH)
                if done:
                    avg_steps_to_success+=len(seq.action_sequence)
                    avg_dev_steps+=(len(seq.action_sequence)-opt_steps)
                    num_success+=1
                    break
            if j==num_paths:
                optimal_path = seq.patch_sequence
            else:
                paths[j] = seq.patch_sequence
                print(paths[j])
            self.buffer.clear()
        print(f"Val Success : {num_success} \t Val Avg Reward : {avg_reward/(total_imgs*5)} \t Val Avg Steps Success : {avg_steps_to_success/(num_success+1e-6)} \t Val Dev : {avg_dev_steps/(num_success+1e-6)}")
        np.save("path_sequence.npy", paths)
        np.save("optimal_sequence.npy", optimal_path)
        return num_success

    def visualize_paths_multimodal(self, valid_path, ground_path, idx, start, goal, num_paths=100):
        sat_dict = np.load(valid_path, allow_pickle=True)
        ground_dict = np.load(ground_path, allow_pickle=True)
        goal_ground = ground_dict[()][f"img_{idx}"]#ground_dict[idx]#
        total_imgs = len(sat_dict[()].keys())
        num_success=0
        avg_dev_steps=0 
        avg_steps_to_success=0
        avg_reward=0
        paths = {}
        optimal_path = []
        GOAL_PATCH = goal
        CURRENT_PATCH = start
        x_min = 5 - GOAL_PATCH%5  
        x_max = x_min + 5
        y_min = 5 - GOAL_PATCH//5
        y_max = y_min + 5
        sat_embeds = sat_dict[()][f"img_{idx}"][y_min:y_max, x_min:x_max].reshape(25, -1)
        for j in range(num_paths+1):

            seq = Sequence(sat_embeds, num_patches=5)
            seq.init_with_goal_embed(goal_ground, GOAL_PATCH)
            seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)
            best_dist=get_dist(GOAL_PATCH, CURRENT_PATCH)
            opt_steps=get_dist(GOAL_PATCH, CURRENT_PATCH)

            for t in (range(1, cfg.train.hparams.max_ep_len+1)):

                inputs = seq.get_input_for_model(device='cuda:0')

                if inputs["actions"] == []:
                    state = self.llm(inputs_embeds=inputs["inputs_embeds"],
                    patch_sequence=inputs["patch_sequence"][:, 1:],
                    patch_size=cfg.data.patch_size)
                else:
                    state = self.llm(
                        inputs_embeds=inputs["inputs_embeds"],
                        actions=[inputs["actions"]],
                        patch_sequence=inputs["patch_sequence"][:, 1:],
                        patch_size=cfg.data.patch_size)
                
                if j==num_paths:
                    action = self.select_greedy_action(state, seq.patch_sequence, cfg.data.patch_size)
                else:
                    action = self.select_stochastic_action(state, seq.patch_sequence, cfg.data.patch_size)

                seq.update_sequence_with_action(action_list[action])
                
                current_patch_id = seq.patch_sequence[-1]
                prev_patch_id = seq.patch_sequence[-2]
                goal_patch_id = seq.patch_sequence[0]

                reward = self.get_reward(cfg.data.patch_size, prev_patch_id, current_patch_id, goal_patch_id, seq.patch_sequence[1:-1], action, best_dist)
                if reward==1:
                    best_dist=get_dist(current_patch_id, goal_patch_id)
                avg_reward+=reward
                done = (current_patch_id==GOAL_PATCH)
                if done:
                    avg_steps_to_success+=len(seq.action_sequence)
                    avg_dev_steps+=(len(seq.action_sequence)-opt_steps)
                    num_success+=1
                    break
            if j==num_paths:
                optimal_path = seq.patch_sequence
            else:
                paths[j] = seq.patch_sequence
                print(paths[j])
            self.buffer.clear()
        print(f"Val Success : {num_success} \t Val Avg Reward : {avg_reward/(total_imgs*5)} \t Val Avg Steps Success : {avg_steps_to_success/(num_success+1e-6)} \t Val Dev : {avg_dev_steps/(num_success+1e-6)}")
        np.save("crop_params.npy", [y_min, y_max, x_min, x_max])
        np.save("path_sequence.npy", paths)
        np.save("optimal_sequence.npy", optimal_path)
        return num_success

