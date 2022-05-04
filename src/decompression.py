
from ._defs import *
import jpeglib
from pathlib import Path
import numpy as np
import tempfile
import pandas as pd


def run_baseline_decompression(vs_decomp: list[str], v_comp: str, dataset: np.array):
    images_rgb = {'comp_version': [], 'decomp_version': [], 'image': []}

    tmp = tempfile.NamedTemporaryFile()  # create temporary file
    for i, v_decomp in enumerate(vs_decomp):

        fnames = [str(Path(tmp) / f'{i}.jpeg')
                  for i in range(dataset.shape[0])]

        # compress each image the fixed default version
        with jpeglib.version(v_comp):
            for i, fname in enumerate(fnames):
                im = jpeglib.from_spatial(dataset[i])
                im.write_spatial(fname)

        # decompress each image with each version
        with jpeglib.version(v_decomp):
            images_rgb['comp_version'].append(v_comp)
            images_rgb['decomp_version'].append(v_decomp)
            images_rgb['image']

    return pd.DataFrame(images_rgb)
