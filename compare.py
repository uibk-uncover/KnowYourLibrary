from math import dist
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform


def mismatch(x1, x2): return (
    np.abs(x1.astype(np.int32) - x2.astype(np.int32)) != 0).mean()


def get_distance_matrix(images_rgb: pd.Dataframe, comp_versions: list[str]) -> np.array:

    images_rgb_list = np.array(
        [list(i) for i in images_rgb.image.to_list()], dtype=object)
    images_rgb_list = images_rgb_list.reshape(len(comp_versions), -1)

    # get pairwise distances between observations in n-dimensional space.
    dists_rgb = pdist(images_rgb_list, mismatch)

    return squareform(dists_rgb)


def compare_rgb_dct_mismatches(images_rgb: np.array):

    dist_matrix_rgb = get_distance_matrix(images_rgb)

    dist_matrix_Y = get_distance_matrix(images_rgb.Y)
    dist_matrix_Cb = get_distance_matrix(images_rgb.Cb)
    dist_matrix_Cr = get_distance_matrix(images_rgb.Cr)

    dist_matrix_DCT = {'Y': dist_matrix_Y,
                       'Cb': dist_matrix_Cb, 'Cr': dist_matrix_Cr}

    for matrix in dist_matrix_DCT:
        if ((dist_matrix_rgb == 0) != (matrix.value == 0)).all():
            print('mismatches in channel', matrix.key)
