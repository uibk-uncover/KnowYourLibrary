
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
    pass

def run_test(dataset: np.ndarray, ctx: TestContext):
    # parse
    N,_,_,channels = dataset.shape
    ctx.colorspace = _cspaces[channels]
    
    # temporary directory
    tmp = tempfile.TemporaryDirectory()
    
    # iterate versions
    images = {'version': [], 'Y': [], 'Cb': [], 'Cr': []}
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
            decompressed = [
                jpeglib.read_spatial(fname)
            ]

