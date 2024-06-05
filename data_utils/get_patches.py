from PIL import Image
import torch
import numpy as np
import os
import fire
import glob
import pandas as pd


def get_patches(img_path, patch_size=5, save_path="patches_images"):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((1500,1500),Image.BICUBIC)
    img_array = torch.from_numpy(np.array(img))

    if img_array.shape[0]%patch_size!=0:
        raise ValueError

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    h, w = img_array.shape[0], img_array.shape[1]
    patch = img_array.unfold(0, h//patch_size, h//patch_size).unfold(1, w//patch_size, w//patch_size).reshape(-1, 3, h//patch_size, w//patch_size).numpy()
    for i in range(patch_size**2):
        Image.fromarray(patch[i, :, :, :].transpose(1, 2, 0)).save(os.path.join(save_path, f"patch_{i}.jpg"))


def get_patches_multiple_images(path="gomaa_geo/data",
                                img_path="gomaa_geo/data/image",
                                csv_path="gomaa_geo/data/list_eval_partition.csv",
                                patch_size=20,
                                partition=2.0):
    df = pd.read_csv(csv_path)
    df = df[df["partition"]==partition]
    for i in range(len(df)):
        get_patches(os.path.join(img_path, df.iloc[i]["image_id"]), save_path=os.path.join(path, f"patches_test_20/img_{i}"), patch_size=patch_size)

def get_patches_directory(path="gomaa_geo/data",
                          img_path="gomaa_geo/data/test/images",
                          patch_size=10):
    l = sorted(glob.glob(os.path.join(img_path, '*.png')))

    print(l[:5])
    for i in range(len(l)):
        get_patches(l[i], save_path=os.path.join(path, f"patches_sat/img_{i}"), patch_size=patch_size)

def get_patches_sat_ground(path="gomaa_geo/data",
                          img_path="gomaa_geo/data/test/images",
                          patch_size=5):
    l = sorted(glob.glob(img_path))

    print(l[:5])
    for i in range(len(l)):
        get_patches(glob.glob(os.path.join(l[i],"stitch*"))[0], save_path=os.path.join(path, f"patches_sat/img_{i}"), patch_size=patch_size)


if __name__=='__main__':
    fire.Fire(get_patches_multiple_images)
