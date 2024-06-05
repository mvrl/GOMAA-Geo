import pandas as pd
from transformers import AutoTokenizer, CLIPTextModelWithProjection
import numpy as np

df = pd.read_csv("SatNav/data/coords_captions.csv")
model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
inputs = tokenizer(df["Caption"].tolist(), padding=True, return_tensors="pt", max_length=77, truncation=True)
outputs = model(**inputs)
text_embeds = outputs.text_embeds

np.save("SatNav/data/papr_my_text_embeds.npy", text_embeds.detach().cpu().numpy())