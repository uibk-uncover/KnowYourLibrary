
import collections
import jpeglib
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from ._defs import *
from . import mismatch


CompressionTestResults = collections.namedtuple(
    'CompressionTestResults', ['Y', 'Cb', 'Cr', 'spatial'])


def run_test(dataset: np.ndarray, ctx: TestContext):
    # parse
    N, _, _, channels = dataset.shape
    ctx.colorspace = cspaces[channels]

    # temporary directory
    tmp = tempfile.TemporaryDirectory()

    # iterate versions
    images = {'version': [], 'spatial': [], 'Y': [], 'Cb': [], 'Cr': []}
    for i, v_compress in enumerate(ctx.versions):

        # compress with each version
        with jpeglib.version(v_compress):
            fnames = [str(Path(tmp.name) / f'{i}.jpeg') for i in range(N)]
            [
                compress_image(dataset[j], fnames[j], ctx)
                for j, fname in enumerate(fnames)
            ]
        # decompress with single (arbitrary) version
        with jpeglib.version(ctx.v_arbitrary):
            dct = [
                read_jpeg(fname, ctx)
                for j, fname in enumerate(fnames)
            ]
            spatial = [
                decompress_image(fname, ctx)
                for j, fname in enumerate(fnames)
            ]
            images['version'].append(v_compress)
            images['spatial'].append(np.array([x.spatial for x in spatial]))
            images['Y'].append(np.array([x.Y for x in dct]))
            images['Cb'].append(np.array([x.Cb for x in dct]))
            images['Cr'].append(np.array([x.Cr for x in dct]))

    # results to dataframe
    images = pd.DataFrame(images)

    # get clusters of equal results
    def _prepare(df, var): return df[['version', var]].rename(
        {var: 'x'}, axis=1)
    clusters_Y = mismatch.clusters(_prepare(images, 'Y'))
    clusters_Cb = mismatch.clusters(_prepare(images, 'Cb'))
    clusters_Cr = mismatch.clusters(_prepare(images, 'Cr'))

    # get clusters from spatial for control
    clusters = mismatch.clusters(_prepare(images, 'spatial'))

    # return
    return CompressionTestResults(clusters_Y, clusters_Cb, clusters_Cr, clusters)


def print_clusters(clusters):
    print("| Y", clusters.Y)
    if mismatch.is_clustering_same(clusters.Cb, clusters.Cr):
        print("| C*", clusters.Cb)
    else:
        print("| Cb", clusters.Cb)
        print("| Cr", clusters.Cr)
    if not mismatch.is_clustering_same(clusters.Cb, clusters.spatial):
        print("| spatial", clusters.spatial)

_joint = collections.OrderedDict()
def _clusters_to_key(clusters):
    return tuple(tuple(i for i in c) for c in clusters)
def _key_to_clusters(key):
    return list(list(i for i in c) for c in clusters)
def add_print_grouped_clusters(clusters, identifier):
    """Call me with clusters and identifier of a call. I will keep track of it and print it nicely for you."""
    global _joint
    # is empty
    was_empty = len(_joint) == 0
    # add to joint
    key = _clusters_to_key(clusters)
    if clusters not in _joint:
        _joint[key] = [identifier]
    else:
        _joint[key].append(identifier)
    # get the first
    k1 = next(iter(_joint))
    print("First is", k1)
    # print legend
    if was_empty:
        print(_key_to_clusters(k1), ":", end="")
    # print until empty
    while _joint[k1]:
        print(" ", _joint[k1][0], sep="", end="")
        _joint[k1] = _joint[k1][1:]
    
def end_print_grouped_clusters():
    """Call me when done with add_print_grouped_clusters."""
    global _joint
    for k in _joint:
        # skip empty (first)
        if not _joint[k]:
            continue
        # print legend
        print(_key_to_clusters(k1), ":", end="")
        # print until empty
        while _joint[k1]:
            print(" ", _joint[k1][0], sep="", end="")
            _joint[k1] = _joint[k1][1:]
