
from math import dist
import re
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from typing import Tuple


def L1(x1, x2):
    return (np.abs(x1.astype(np.int32) - x2.astype(np.int32)) != 0).mean()


def get_clusters(images: pd.DataFrame) -> Tuple[Tuple[str]]:
    """Call me to get clusters from the dataframe.

    Parameters:
        images (pd.DataFrame): Must have columns "version" and "x".
            version (str): identifier of version
            x (np.array): spatial or DCT tensor
    Returns:
        (list): Returns list of sets, representing the clusters.
    """
    # get version list
    versions = images.version.unique().tolist()

    # flatten images per version
    images_flattened = (
        np.array([
            list(i) for i in images.x.to_list()
        ], dtype=object)
        .reshape(len(versions), -1)
    )
    # get pairwise L1 distances between versions
    distmat = squareform(pdist(images_flattened, L1))

    # cluster
    for k in range(1, len(versions)+1):
        agnes = AgglomerativeClustering(
            n_clusters=k, linkage='average', affinity='precomputed')
        agnes.fit(distmat)

        # compute heterogenity metric (sum of distances)
        heterogenity = np.sum([distmat[i, j]
                               for group in np.unique(agnes.labels_)
                               for i in np.where(agnes.labels_ == group)[0]
                               for j in np.where(agnes.labels_ == group)[0]])

        # keep homogenous clusters
        if heterogenity == 0:
            break
    clusters = tuple(tuple(versions[i] for i in np.where(
        agnes.labels_ == cl)[0]) for cl in np.unique(agnes.labels_))
    return order_cluster_group(clusters)


def is_clustering_same(c1, c2):
    c1 = {tuple(i for i in sorted(c)) for c in c1}
    c2 = {tuple(i for i in sorted(c)) for c in c2}
    return all(c in c2 for c in c1) and all(c in c1 for c in c2)


def order_cluster_group(cluster_group: Tuple[Tuple[str]]) -> Tuple[Tuple[str]]:
    string_cluster_group = tuple(str(cluster) for cluster in cluster_group)
    order = np.argsort(string_cluster_group)
    return tuple(cluster_group[o] for o in order)
