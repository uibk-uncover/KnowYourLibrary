
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
    clusters_Y = mismatch.get_clusters(_prepare(images, 'Y'))
    clusters_Cb = mismatch.get_clusters(_prepare(images, 'Cb'))
    clusters_Cr = mismatch.get_clusters(_prepare(images, 'Cr'))

    # get clusters from spatial for control
    clusters = mismatch.get_clusters(_prepare(images, 'spatial'))

    # return
    return CompressionTestResults(clusters_Y, clusters_Cb, clusters_Cr, clusters)
