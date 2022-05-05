
import numpy as np
import jpeglib
from typing import Tuple


class TestContext:
    # versions to test
    versions: list = ['6b', 'turbo210', '7', '8', '8a',
                      '8b', '8c', '8d', '9', '9a', '9b', '9c', '9d', '9e']
    # arbitrary version
    v_arbitrary: str = '9e'
    # chroma subsampling
    samp_factor: Tuple[Tuple[int, int],
                       Tuple[int, int], Tuple[int, int]] = None
    use_fancy_sampling: bool = None
    # DCT method
    dct_method_compression: str = None
    dct_method_decompression: str = None
    # quality
    quality: int = None
    # colorspace
    colorspace: str = None
    # compressor function
    compressor = None


cspaces = {1: 'JCS_GRAYSCALE', 3: 'JCS_RGB'}
_flags = {True: ['+DO_FANCY_UPSAMPLING'],
          False: ['-DO_FANCY_UPSAMPLING'], None: []}


def compress_image(x: np.ndarray, path: str, ctx: TestContext):
    # to jpeglib
    im = jpeglib.from_spatial(x)
    # samp factor
    if ctx.samp_factor is not None:
        im.samp_factor = ctx.samp_factor
    # chroma subsampling
    flags = _flags[ctx.use_fancy_sampling]
    # compress
    im.write_spatial(path, qt=ctx.quality,
                     dct_method=ctx.dct_method_compression, flags=flags)


def decompress_image(path: str, ctx: TestContext):
    # chroma subsampling
    flags = _flags[ctx.use_fancy_sampling]
    # decompress
    return jpeglib.read_spatial(path, dct_method=ctx.dct_method_decompression, flags=flags)


def read_jpeg(path: str, ctx: TestContext):
    # read DCT
    return jpeglib.read_dct(path)
