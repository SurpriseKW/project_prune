from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np


def get_num_clusters(layer, matrix_info_retention_ratio):
    ts = layer.weight.data.clone().cpu()
    ts = ts.reshape(ts.size(0), -1)
    mat = np.array(ts).T

    pca = PCA(n_components=matrix_info_retention_ratio).fit(mat)
    mat_ = pca.transform(mat)
    return mat_.shape[1]


def get_remain_channel(layer, matrix_info_retention_ratio):
    ts = layer.weight.data.clone().cpu()
    ts = ts.reshape(ts.size(0), -1)
    mat = np.array(ts)

    num_clusters = get_num_clusters(layer, matrix_info_retention_ratio) if matrix_info_retention_ratio < 1 else matrix_info_retention_ratio
    pre = KMeans(n_clusters=num_clusters).fit_predict(mat)

    visited_clusters = set()
    remain_channel = []
    for i in range(len(pre)):
        if pre[i] not in visited_clusters:
            remain_channel.append(i)
            visited_clusters.add(pre[i])
    return remain_channel



