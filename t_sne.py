import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib
import seaborn as sns
import pandas as pd


def main():
    # get_lr_kpts()
    data = np.load("knns/promptir/lr_features_6.npy")
    for i in range(data.shape[0]):
        data[i, ...] = data[i, ...] / np.linalg.norm(data[i, ...])
    label = np.load("knns/promptir/lr_labels.npy")
    # data = np.load("kpts_nonoffset.npy")
    # label = np.load("labels_nonoffset.npy")
    # data, label, n_samples, n_features = get_data()
    print(label.shape, data.shape)
    # del data, label
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, verbose=1, n_jobs=4, n_iter=2000)
    # t0 = time()
    tsne_results = tsne.fit_transform(data)
    # feat_cols = ["super-resolution", "haze", "motion-blur", "noise", "rain"]
    feat_cols = ['feature_'+str(i) for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=feat_cols)
    label_degradation = np.where(label==1, "haze", label)
    label_degradation = np.where(label==2, "motion-blur", label_degradation)
    label_degradation = np.where(label==3, "noise", label_degradation)
    label_degradation = np.where(label==4, "rain", label_degradation)
    label_degradation = np.where(label==5, "low_light", label_degradation)
    # print(label_degradation)
    df['degradation'] = label_degradation
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    plt.figure(figsize=(16,10))
    plt.xticks([])
    plt.yticks([])
    matplotlib.rcParams.update({'font.size': 24})
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="degradation",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        s=128,
        # alpha=0.3,
    )
    plt.title("KL divergence after 2000 iterations: 0.814725.")
    plt.savefig("knns/t_sne_promptir.png",bbox_inches='tight',dpi=300,pad_inches=0.0)

main()
