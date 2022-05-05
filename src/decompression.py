
from ._defs import *
import jpeglib
import collections
from pathlib import Path
import numpy as np
import tempfile
import pandas as pd
from . import mismatch

DecompressionTestResults = collections.namedtuple(
    'DecompressionTestResults', ['spatial'])


def run_test(dataset: np.ndarray, ctx: TestContext()) -> pd.DataFrame:
    sample_size, _, _, channels = dataset.shape
    ctx.colorspace = cspaces[channels]

    images_rgb = {'comp_version': [], 'version': [], 'spatial': []}

    tmp = tempfile.TemporaryDirectory()  # create temporary directory
    for i, v_decomp in enumerate(ctx.versions):

        fnames = [str(Path(tmp.name) / f'{i}.jpeg')
                  for i in range(sample_size)]

        # compress each image the fixed default version
        with jpeglib.version(ctx.v_arbitrary):
            [
                compress_image(dataset[j], fnames[j], ctx)
                for j, fname in enumerate(fnames)
            ]

        # decompress each image with each version
        with jpeglib.version(v_decomp):

            spatial = [
                decompress_image(fname, ctx)
                for j, fname in enumerate(fnames)
            ]

            images_rgb['comp_version'].append(ctx.v_arbitrary)
            images_rgb['version'].append(v_decomp)
            images_rgb['spatial'].append(
                np.array([x.spatial for x in spatial]))

    images = pd.DataFrame(images_rgb)

    # get clusters of equal results
    def _prepare(df, var): return df[['version', var]].rename(
        {var: 'x'}, axis=1)

    clusters = mismatch.get_clusters(_prepare(images, 'spatial'))

    return DecompressionTestResults(clusters)
