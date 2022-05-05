
import collections
import jpeglib
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys
from ._defs import *
from . import mismatch


CompressionTestResults = collections.namedtuple(
    'CompressionTestResults', ['Y', 'Cb', 'Cr', 'spatial'])

def run_test(dataset: np.ndarray, ctx: TestContext):
    """Executes compression test.
    
    Args:
        dataset (np.ndarray): 4D tensor with dataset
        ctx (TestContext): context for the test
    """
    # parse
    N, _, _, channels = dataset.shape
    ctx.colorspace = cspaces[channels]

    # temporary directory
    tmp = tempfile.TemporaryDirectory()

    # iterate versions
    if channels == 3:
        images = {'version': [], 'spatial': [], 'Y': [], 'Cb': [], 'Cr': []}
    else:
        images = {'version': [], 'spatial': [], 'Y': []}
    for i, v_compress in enumerate(ctx.versions):
        fnames = [str(Path(tmp.name) / f'{i}.jpeg') for i in range(N)]

        # compress with each version
        if not ctx.compressor:
            with jpeglib.version(v_compress):        
                [
                    compress_image(dataset[j], fnames[j], ctx)
                    for j, fname in enumerate(fnames)
                ]
        else:
            ctx.compressor(dataset, fnames, v_compress, ctx)

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
            if channels == 3:
                images['Cb'].append(np.array([x.Cb for x in dct]))
                images['Cr'].append(np.array([x.Cr for x in dct]))
    
    # results to dataframe
    images = pd.DataFrame(images)

    # get clusters of equal results
    def _prepare(df, var): return df[['version', var]].rename(
        {var: 'x'}, axis=1)
    clusters_Y = mismatch.get_clusters(_prepare(images, 'Y'))
    if channels == 3:
        clusters_Cb = mismatch.get_clusters(_prepare(images, 'Cb'))
        clusters_Cr = mismatch.get_clusters(_prepare(images, 'Cr'))
    else:
        clusters_Cb = clusters_Cr = None

    # get clusters from spatial for control
    clusters = mismatch.get_clusters(_prepare(images, 'spatial'))

    # return
    return CompressionTestResults(clusters_Y, clusters_Cb, clusters_Cr, clusters)

def print_clusters(clusters):
    print("| Y", clusters.Y)
    # rgb
    if clusters.Cb is not None and clusters.Cr is not None:
        if mismatch.is_clustering_same(clusters.Cb, clusters.Cr):
            print("| C*", clusters.Cb)
        else:
            print("| Cb", clusters.Cb)
            print("| Cr", clusters.Cr)
        if not mismatch.is_clustering_same(clusters.Cb, clusters.spatial):
            print("| spatial", clusters.spatial)
    # grayscale
    else:
        if not mismatch.is_clustering_same(clusters.Y, clusters.spatial):
            print("| spatial", clusters.spatial)

def _to_key(k):
    pass
    

_joint = collections.OrderedDict()
def add_print_grouped_clusters(clusters, identifier):
    """Call me with clusters and identifier of a call. I will keep track of it and print it nicely for you."""
    global _joint
    # is empty
    was_empty = len(_joint) == 0
    # choose channels to add
    channels = {'Y': clusters.Y, 'spatial': clusters.spatial}
    if clusters.Cb is not None and clusters.Cr is not None:
        if clusters.Cb == clusters.Cr:
            if clusters.Y != clusters.Cb:
                channels['C*'] = clusters.Cb
        else:
            if clusters.Y != clusters.Cb:
                channels['Cb'] = clusters.Cb
            if clusters.Y != clusters.Cr:
                channels['Cr'] = clusters.Cr
    for c in ['Y','C*','Cb','Cr','spatial']:
        # do not print
        if c not in channels:
            continue
        # get channel clustering
        clust = channels[c]
        clust = _to_key(clust)
        # create channel identifier
        identifier_c = f'{identifier}:{c}'
        # add to joint
        if clust not in _joint:
            _joint[clust] = [identifier_c]
        else:
            _joint[clust].append(identifier_c)
    # get the first
    k1 = next(iter(_joint))
    # print legend
    if was_empty:
        print("|", k1, ":", end="")
    # print until empty
    while _joint[k1]:
        print(" ", _joint[k1][0], sep="", end="")
        _joint[k1] = _joint[k1][1:]
    sys.stdout.flush()
    
def end_print_grouped_clusters():
    """Call me when done with add_print_grouped_clusters."""
    global _joint
    for k in _joint:
        # skip empty (first)
        if not _joint[k]:
            print()
            continue
        # print legend
        print("|", k, ":", end="")
        # print until empty
        while _joint[k]:
            print(" ", _joint[k][0], sep="", end="")
            _joint[k] = _joint[k][1:]
        print()
    _joint = collections.OrderedDict()
