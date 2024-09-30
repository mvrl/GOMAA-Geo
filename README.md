# GOMAA-Geo
PyTorch implementation of _GOMAA-Geo: GOal Modality Agnostic Active Geo-localization_ (NeurIPS 2024)

<div align="center">
<img src="imgs/logo-3.png" width="350">

[![arXiv](https://img.shields.io/badge/arXiv-2406.01917-red
)](https://arxiv.org/abs/2406.01917v1)
[![Project Page](https://img.shields.io/badge/Project-Website-green)]()
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Spaces-yellow?style=flat&logo=hug)]()</center>

[Anindya Sarkar*](https://scholar.google.co.in/citations?user=2hQyYz0AAAAJ&hl=en),
[Srikumar Sastry*](https://sites.wustl.edu/srikumarsastry/),
[Aleksis Pirinen](https://aleksispi.github.io/),
[Chongjie Zhang](https://engineering.wustl.edu/faculty/Chongjie-Zhang.html),
[Nathan Jacobs](https://jacobsn.github.io/),
[Yevgeniy Vorobeychik](https://vorobeychik.com/)
(*Corresponding Author)
</div>

![gomaa-thumbnail](https://github.com/user-attachments/assets/1a3288e8-606e-4336-b567-d98a17bde487)

This repository is the official implementation of the **NeurIPS 2024 paper** [_GOMAA-Geo_](https://arxiv.org/abs/2406.01917v1), a goal modality agnostic active geo-localization agent that can geo-localize a goal location -- specified as an aerial patch, ground-level image, or textual description -- by navigating partially observed aerial imagery.

![](imgs/teaser_v2.jpg)

## ⏭️ Next
- [ ] Update Gradio demo
- [ ] Release Models to 🤗 HuggingFace
- [ ] Release PyTorch `ckpt` files for all models

## 🎬 Installation

You can use the following commands to install the necessary dependencies to run the code:
```bash
conda create --name gomaa_geo
conda activate gomaa_geo
conda install python==3.11
pip install -r requirements.txt
```

## ⬇️ Getting the data

To run the code with the Masa or xBD data, download the zip file at the following link: https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset 
To run the code with the xBD data, download the zip file at the following link: https://xview2.org/ (Note that, In order to download the dataset, first login using a valid email id) 
To run the code with our MM-GAG data, download the zip file at the following Anonymous Huggingface link: https://huggingface.co/datasets/Gomaa-Geo/MM-GAG

The uncompressed folder named `data` should be placed at the root directory of this repository.
This folder includes processed data for the following active geo localization problems:
- Masa dataset in `masa_data`, xBD dataset in 'xBD_data'

This folder also includes other intermediate results to recreate our analyses and figures.

## 📄 Specify all configurations
Before setting up data or running experiments, setup all parameters of interest in the file `config.py`. This includes grid size, model configuration, training configuration etc.

## 📀 Process the Data
Extract all data of interest in the folder `gomaa_geo/data`

Then, create patches (grids) for each image in a dataset using the following script:
```bash
python -m gamma_geo.data_utils.get_patches
```
Then get CLIP-MMFE embeddings for each patch:
```bash
python -m gamma_geo.data_utils.get_sat_embeddings_sat2cap
```

Using the same script and function `get_ground_embeds` one can create embeddings for ground level images.

To create text embeddings, run the following script:
```bash
python -m gamma_geo.data_utils.get_text_embeddings
```

## 🔥 Running the code

To run our pre-training procedure with GOMAA-Geo, use the following commands:
```bash
python -m gomaa_geo.pretrain
```
Again, all parameters of interest must be specified in the `config.py` file.

The weights of the trained llm network at each iteration will be locally saved in `gomaa_geo/checkpoint/`

To run training of the pre-trained model, use the following command:
```bash
python -m gomaa_geo.train
```

To run inference, run the following command:
```bash
python -m gomaa_geo.validate
```
Set the path to the pre-trained llm in the variable: `cfg.train.llm_checkpoint`

To visualize exploration behaviour of the trained model, run the following script:
```bash
python -m gomaa_geo.viz_path --idx=77 --start=0 --end=24
```
where, `idx` is image id, `start` is the starting position and `end` is the goal position.

## 🐨 Model Zoo
Download GOMAA-Geo models from the links below:
Coming Soon ...


## 📑 Citation

```bibtex
@article{sarkar2024gomaa,
  title={GOMAA-Geo: GOal Modality Agnostic Active Geo-localization},
  author={Sarkar, Anindya and Sastry, Srikumar and Pirinen, Aleksis and Zhang, Chongjie and Jacobs, Nathan and Vorobeychik, Yevgeniy},
  journal={arXiv preprint arXiv:2406.01917},
  year={2024}
}
```

## 🔍 Additional Links
Check out our lab website for other interesting works on geospatial understanding and mapping:
* Multi-Modal Vision Research Lab (MVRL) - [Link](https://mvrl.cse.wustl.edu/)
* Related Works from MVRL - [Link](https://mvrl.cse.wustl.edu/publications/)

