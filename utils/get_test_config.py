import numpy as np
from ..data_utils import SequenceDummy, Sequence
from ..config import cfg
from transformers import AutoTokenizer
from collections import defaultdict
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.distributions import Categorical

def gaussian_kernel(size: int, sigma: float):
    """Creates a 2D Gaussian kernel."""
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2)/(2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)

def create_gaussian_kernel_2d(size: int, sigma: float):
    """Creates a 2D Gaussian kernel with the specified size and standard deviation."""
    kernel = gaussian_kernel(size, sigma)
    # Convert numpy array to torch tensor
    kernel = torch.FloatTensor(kernel)
    # Add dimensions to match the expected shape for convolution
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return kernel

# Define size and sigma for the Gaussian kernel
kernel_size = 5
sigma = 1.0
print(f"{sigma=}")
# Create the Gaussian kernel
gaussian_kernel_2d = create_gaussian_kernel_2d(kernel_size, sigma)
cat_dist = Categorical(gaussian_kernel_2d.view(-1))

action_list = {0:"up",
1:"right",
2:"down",
3:"left"
}

direction_list = {}
dir_class_id = 0
for i in range(-4, 5):
    for j in range(-4, 5):
        angle = np.round(np.arctan(i/(j+np.finfo(float).eps)), 6)
        if (i<0 and j<0) or (i<0 and j>0):
            angle+=np.pi
        if i==0 and j==0:
            angle = 0
        if angle not in direction_list:
            direction_list[angle] = dir_class_id
            dir_class_id+=1

def get_optimal_actions(goal_patch, cur_patch, patch_size=5):
    p1_rows = goal_patch//patch_size
    p1_cols = goal_patch%patch_size

    p2_rows = cur_patch//patch_size
    p2_cols = cur_patch%patch_size

    opt_actions = [0]*4

    if p1_rows < p2_rows:
        opt_actions[0] = 1
    if p1_rows > p2_rows:
        opt_actions[2] = 1
    if p1_cols > p2_cols:
        opt_actions[1] = 1
    if p1_cols < p2_cols:
        opt_actions[3] = 1
    return opt_actions

def get_dist(p1, p2, patch_size=5):
    p1_rows = p1//patch_size
    p1_cols = p1%patch_size

    p2_rows = p2//patch_size
    p2_cols = p2%patch_size

    return abs(p1_rows-p2_rows) + abs(p1_cols-p2_cols)

def generate_config(data_path, patch_size=5, n_config_per_img=5, dist=4):

    config = defaultdict(list)
    data = np.load(data_path, allow_pickle=True)
    for i in range(len(data[()].keys())):
        for j in range(n_config_per_img):
            CURRENT_PATCH = np.random.randint(0, patch_size**2)
            GOAL_PATCH = np.random.randint(0, patch_size**2)
            while get_dist(CURRENT_PATCH, GOAL_PATCH) != dist:
                CURRENT_PATCH = np.random.randint(0, patch_size**2)
                GOAL_PATCH = np.random.randint(0, patch_size**2)
            
            config[f'img_{i}'].append((GOAL_PATCH, CURRENT_PATCH))

    return config

def generate_random_dist_config(data_path, patch_size=5, n_config_per_img=5, dist_possible = [1, 2]):

    config = defaultdict(list)
    data = np.load(data_path, allow_pickle=True)
    for i in range(len(data[()].keys())):
        for j in range(n_config_per_img):
            dist = np.random.randint(dist_possible[0], dist_possible[1]+1)
            CURRENT_PATCH = np.random.randint(0, patch_size**2)
            GOAL_PATCH = np.random.randint(0, patch_size**2)
            while get_dist(CURRENT_PATCH, GOAL_PATCH) != dist:
                CURRENT_PATCH = np.random.randint(0, patch_size**2)
                GOAL_PATCH = np.random.randint(0, patch_size**2)
            config[f'img_{i}'].append((GOAL_PATCH, CURRENT_PATCH))

    return config


def get_random_sequence(embeds, patch_size, sequence_length=10):
    seq = Sequence(embeds, num_patches=patch_size)

    dist = np.random.randint(1, 9)
    GOAL_PATCH = np.random.randint(0, cfg.data.patch_size**2)
    CURRENT_PATCH = np.random.randint(0, cfg.data.patch_size**2)
    while get_dist(CURRENT_PATCH, GOAL_PATCH) != dist:#GOAL_PATCH == CURRENT_PATCH:
        GOAL_PATCH = np.random.randint(0, cfg.data.patch_size**2)
        CURRENT_PATCH = np.random.randint(0, cfg.data.patch_size**2)
    seq.init_with_goal_image(GOAL_PATCH)
    seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)
    dist = []
    dist.append(get_dist(GOAL_PATCH, CURRENT_PATCH))
    act = []
    opt_actions = []
    opt_actions.append(get_optimal_actions(GOAL_PATCH, CURRENT_PATCH))
    for i in range(sequence_length):
        action = np.random.randint(0, 4)
        act.append(action)
        seq.update_sequence_with_action(action_list[action])
        dist.append(get_dist(GOAL_PATCH, seq.patch_sequence[-1]))
        opt_actions.append(get_optimal_actions(GOAL_PATCH, seq.patch_sequence[-1]))
    return seq, act, dist, opt_actions


def get_pretrain_sequence(embeds, patch_size, min_length=6):
    seq = Sequence(embeds, num_patches=patch_size)
    dir_vectors = []
    GOAL_PATCH = np.random.randint(0, patch_size**2)
    CURRENT_PATCH = np.random.randint(0, patch_size**2)
    while get_dist(GOAL_PATCH, CURRENT_PATCH)<min_length: 
        GOAL_PATCH = np.random.randint(0, patch_size**2)
        CURRENT_PATCH = np.random.randint(0, patch_size**2)
    
    seq.init_with_goal_image(GOAL_PATCH)
    seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)

    p1_rows = CURRENT_PATCH//patch_size
    p1_cols = CURRENT_PATCH%patch_size

    p2_rows = GOAL_PATCH//patch_size
    p2_cols = GOAL_PATCH%patch_size

    while seq.patch_sequence[-1]!=GOAL_PATCH:

        i = (p2_cols-p1_cols)
        j = (p2_rows-p1_rows)
        angle = np.round(np.arctan(i/(j+np.finfo(float).eps)), 6)
        if (i<0 and j<0) or (i<0 and j>0):
            angle+=np.pi
        if i==0 and j==0:
            angle = 0
        dir_vectors.append(direction_list[angle])

        if abs(p2_rows - p1_rows) > abs(p2_cols - p1_cols):
            if p2_rows < p1_rows:
                seq.update_sequence_with_action('up')
            else:
                seq.update_sequence_with_action('down')
        elif abs(p2_rows - p1_rows) < abs(p2_cols - p1_cols):
            if p2_cols < p1_cols:
                seq.update_sequence_with_action('left')
            else:
                seq.update_sequence_with_action('right')
        
        else:
            if p2_rows < p1_rows:
                seq.update_sequence_with_action('up')
            elif p2_cols > p1_cols:
                seq.update_sequence_with_action('right')
            elif p2_rows > p1_rows:
                seq.update_sequence_with_action('down')
            else:
                seq.update_sequence_with_action('left')

        p1_rows = seq.patch_sequence[-1]//patch_size
        p1_cols = seq.patch_sequence[-1]%patch_size

    return seq, dir_vectors


class PretrainSequences(Dataset):
    def __init__(self, dataset_dict, patch_size, min_length=6):
        self.dataset_dict = dataset_dict
        self.patch_size = patch_size
        self.min_length = min_length
        self.action_list = {"up": 0,
                            "right": 1,
                            "down": 2,
                            "left": 3
                            }
    
    def __len__(self):
        return len(self.dataset_dict[()].keys())
    
    def __getitem__(self, idx):
        embeds = self.dataset_dict[()][f'img_{idx}']
        seq, dir_vectors = get_pretrain_sequence(embeds, self.patch_size, self.min_length)
        mask = torch.zeros(2*len(seq.patch_sequence)-2)
        mask[0] = 1
        mask[1] = 1
        mask[-2] = 1
        #mask[-1] = 1

        actions = [seq.action_sequence]
        action_list = [[self.action_list[actions[i][j]]
                        for j in range(len(actions[i]))] for i in range(len(actions))]
        return torch.FloatTensor(np.array(seq.embedding_sequence)), torch.tensor(dir_vectors), mask, seq.action_sequence, torch.LongTensor(seq.patch_sequence), torch.tensor(action_list).squeeze(0)


class DTSequences():
    def __init__(self, dataset_dict, patch_size, min_length=6):
        self.dataset_dict = dataset_dict
        self.patch_size = patch_size
        self.min_length = min_length
        self.action_list = {"up": 0,
                            "right": 1,
                            "down": 2,
                            "left": 3
                            }
    
    def __call__(self):
        embeds = self.dataset_dict
        seq, *_ = get_pretrain_sequence(embeds, self.patch_size, self.min_length)

        actions = [seq.action_sequence]
        action_list = []
        for a in seq.action_sequence:
            one_hot = [0]*5
            one_hot[self.action_list[a]] = 1
            action_list.append(one_hot)
        
        action_list.append([0, 0, 0, 0, 1])
        
        rewards = torch.zeros(len(seq.patch_sequence)-1, 1)
        rewards[-1, 0] = 5
        timesteps = torch.LongTensor(torch.arange(len(seq.patch_sequence)-1))
        r2go = torch.FloatTensor(torch.arange(len(seq.patch_sequence)-2, -1, -1).float()).unsqueeze(1)
        return torch.FloatTensor(np.array(seq.embedding_sequence)[1:]).unsqueeze(0), torch.FloatTensor(action_list).unsqueeze(0), torch.FloatTensor(rewards).unsqueeze(0), r2go.unsqueeze(0), timesteps.unsqueeze(0), torch.LongTensor(seq.patch_sequence[1:]).unsqueeze(0)


class PretrainRandomSequences(Dataset):
    def __init__(self, dataset_dict, patch_size, min_length=6):
        self.dataset_dict = dataset_dict
        self.patch_size = patch_size
        self.min_length = min_length
        self.action_list = {"up": 0,
                            "right": 1,
                            "down": 2,
                            "left": 3
                            }
    
    def __len__(self):
        return len(self.dataset_dict[()].keys())
    
    def __getitem__(self, idx):
        embeds = self.dataset_dict[()][f'img_{idx}']
        seq, act, dist, opt_actions = get_random_sequence(embeds, self.patch_size, np.random.randint(1, 11))
        return torch.FloatTensor(np.array(seq.embedding_sequence)), seq.action_sequence, torch.LongTensor(seq.patch_sequence), torch.tensor(opt_actions)


if __name__=='__main__':

    dataset_dict = np.load(cfg.data.train_path, allow_pickle=True)
    out = get_random_sequence(dataset_dict[()]['img_0'], 5)
    print(out[0].patch_sequence, out[-1])
