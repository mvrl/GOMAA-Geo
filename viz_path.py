import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import fire
import pandas as pd
from copy import deepcopy
import glob
import torch
from .models import PPO
from .utils import seed_everything
from .config import cfg
from copy import deepcopy


def predict(idx=50, start=0, end=24, num_paths=4):
    seed_everything(42)

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

    ## Mass and XbD ##
    ppo_agent.visualize_paths(valid_path, idx=idx, start=start, goal=end, num_paths=num_paths)

    ## Multi Modal: ground##
    #ppo_agent.visualize_paths_multimodal(valid_path, ground_path, idx=idx, start=start, goal=end, num_paths=num_paths)

    ## Multi Modal: text##
    #ppo_agent.visualize_paths_multimodal(valid_path, ground_text_path, idx=idx, start=start, goal=end, num_paths=num_paths)

    main(idx, start, end)

def main(idx=50, start=0, end=24):
    df = pd.read_csv("gomaa_geo/data/list_eval_partition.csv")
    idx = df[df["partition"]==2.0].iloc[idx]['image_id']
    a = np.load("path_sequence.npy", allow_pickle=True)
    c = np.load("optimal_sequence.npy", allow_pickle=True)[1:]
    image_list = list(sorted(glob.glob("gomaa_geo/data/sat_images/*")))
    plt.imshow(np.array(Image.open(image_list[idx])), extent=(-0.5, 4.5, -0.5, 4.5))
    print(c)

    for i in range(4):
        plt.plot([0.5+i, 0.5+i], [-0.5, 4.5], color='1.0', linestyle='--', alpha=0.5)
    
    for i in range(4):
        plt.plot([-0.5,4.5], [ 0.5+i, 0.5+i], color='1.0', linestyle='--', alpha=0.5)

    color = ['w', 'g', 'y', 'c']
    it = iter(color)
    offset = iter([-0.05, -0.1, 0.05, 0.1])
    for key in a[()]:
        b = np.array(a[()][key][1:])
        print(b.shape)
        y = list(4 - b//5)
        x = list(b%5)
        ycopy = deepcopy(y)
        xcopy = deepcopy(x)
        off = next(offset)
        for j in range(len(y)):
            if j<len(y)-1:
                if y[j] == y[j+1]:
                    y[j] = y[j] + off
                    if j>0:
                        if xcopy[j]==xcopy[j-1]:
                            x[j] = x[j-1]
                else:
                    if y[j] > y[j+1]:
                        x[j] = x[j] + off
                    else:
                        x[j] = x[j] - off
                    if j>0:
                        if ycopy[j]==ycopy[j-1]:
                            y[j] = y[j-1]
            else:
                if y[j] == ycopy[j-1]:
                    y[j] = y[j-1]# + off
                else:
                    if y[j] > ycopy[j-1]:
                        x[j] = x[j] - off
                    else:
                        x[j] = x[j] + off
        print(x, y)
        plt.plot(x, y, c=next(it), marker='s')

    plt.plot(c%5, 4-c//5, c='r', marker='*')

    plt.scatter([start%5], [4-start//5], s=100, c='orange', alpha=1.0)
    plt.scatter([end%5], [4-end//5], s=100, c='b', alpha=1.0)
    plt.axis('off')
    plt.tight_layout()

    plt.savefig("path.jpg", dpi=300)

if __name__=='__main__':
    fire.Fire(predict)