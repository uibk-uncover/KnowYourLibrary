
import jpeglib
import numpy as np
import pandas as pd
import tempfile
from ._defs import *

_cspaces = {1: 'JCS_GRAYSCALE', 3: 'JCS_RGB'}
_flags = {True: ['+DO_FANCY_UPSAMPLING'], False: ['-DO_FANCY_UPSAMPLING'], None: []}

def _compress_image(x: np.ndarray, path: str, ctx: TestContext):
    # to jpeglib
    im = jpeglib.from_spatial(x, ctx.colorspace)
    # samp factor
    if ctx.samp_factor is not None:
        im.samp_factor = ctx.samp_factor
    # chroma subsampling
    flags = _flags[ctx.use_chroma_sampling]
    # compress
    im.write_spatial(path, qt=ctx.quality, dct_method=ctx.dct_method, flags=flags)

def _decompress_image(path: str, ctx: TestContext):
    # chroma subsampling
    flags = _flags[ctx.use_chroma_sampling]
    # decompress
    return jpeglib.read_spatial(path, dct_method=ctx.dct_method_arbitrary, flags=flags)

def _read_jpeg(path: str, ctx: TestContext):
    # read DCT
    return jpeglib.read_dct(path)

def run_test(dataset: np.ndarray, ctx: TestContext):
    # parse
    N,_,_,channels = dataset.shape
    ctx.colorspace = _cspaces[channels]
    
    # temporary directory
    tmp = tempfile.TemporaryDirectory()
    
    # iterate versions
    images = {'version': [], 'spatial': [], 'Y': [], 'Cb': [], 'Cr': []}
    for i,v_compress in enumerate(ctx.versions):
        
        # compress with each version
        with jpeglib.version(v_compress):
            fnames = [str(Path(tmp) / f'{i}.jpeg') for i in range(N)]
            [
                _compress_image(dataset[j], fnames[j], ctx)
                for j,fname in enumerate(fnames)
            ]
        
        # decompress with single (arbitrary) version
        with jpeglib.version(ctx.v_arbitrary):
            dct = [
                _read_jpeg(fname, ctx)
                for j,fname in enumerate(fnames)
            ]
            spatial = [
                _decompress_image(fname, ctx)
                for j,fname in enumerate(fnames)
            ]
            images['version'].append(v_compress)
            images['spatial'].append( np.array([x.spatial for x in spatial]) )
            images['Y'].append( np.array([x.Y for x in dct]) )
            images['Cb'].append( np.array([x.Cb for x in dct]) )
            images['Cr'].append( np.array([x.Cr for x in dct]) )
    
    # return dataframe
    return pd.DataFrame(images)
    
            
            

