import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def pca_plot(data_file="gomaa_geo/data/papr_train_sat_embeds_grid_5.npy", patch_size=5, model_name="sat2cap"):

    a = np.load(data_file, allow_pickle=True)

    pca = PCA(n_components=3)

    x = pca.fit_transform(a[()]['img_0'])

    x1 = np.arange(0, patch_size)

    y1 = np.arange(0, patch_size)

    x1, y1 = np.meshgrid(x1, y1)

    plt.scatter(x1.reshape(-1), y1.reshape(-1), c=(x-x.min())/(x.max()-x.min()))

    plt.savefig(f"pca_{model_name}_{patch_size}.jpg")


if __name__=='__main__':
    import fire
    fire.Fire(pca_plot)