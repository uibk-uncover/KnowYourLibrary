
import jpeglib
import numpy as np
import tempfile
from typing import Tuple
from src import implementation
import sys

# sampling factor
samp_factors = [
    ((1, 1), (1, 1), (1, 1)),  # 4:4:4
    ((1, 2), (1, 2), (1, 2)),
    ((2, 1), (2, 1), (2, 1)),

    ((1, 2), (1, 1), (1, 1)),  # 4:4:0
    ((2, 2), (2, 1), (2, 1)),
    ((1, 4), (1, 2), (1, 2)),
    ((1, 2), (1, 2), (1, 1)),   # Cb 4:4:4 Cr 4:4:0
    ((1, 2), (1, 1), (1, 2)),   # Cb 4:4:0 Cr 4:4:4

    ((2, 1), (1, 1), (1, 1)),  # 4:2:2
    ((2, 2), (1, 2), (1, 2)),
    ((2, 1), (2, 1), (1, 1)),   # Cb 4:4:4 Cr 4:2:2
    ((2, 1), (1, 1), (2, 1)),   # Cb 4:2:2 Cr 4:4:4

    ((2, 2), (1, 1), (1, 1)),  # 4:2:0
    ((2, 2), (2, 1), (1, 1)),   # Cb 4:4:0 Cr 4:2:0
    ((2, 2), (1, 1), (2, 1)),   # Cb 4:2:0 Cr 4:4:0
    ((2, 2), (1, 2), (1, 1)),   # Cb 4:2:2 Cr 4:2:0
    ((2, 2), (1, 1), (1, 2)),   # Cb 4:2:0 Cr 4:2:2
    ((2, 2), (2, 2), (1, 1)),   # Cb 4:4:4 Cr 4:2:0
    ((2, 2), (2, 2), (2, 1)),   # Cb 4:4:4 Cr 4:4:0
    ((2, 2), (2, 2), (1, 2)),   # Cb 4:4:4 Cr 4:2:2
    ((2, 2), (1, 1), (2, 2)),   # Cb 4:2:0 Cr 4:4:4
    ((2, 2), (2, 1), (2, 2)),   # Cb 4:4:0 Cr 4:4:4
    ((2, 2), (1, 2), (2, 2)),   # Cb 4:2:2 Cr 4:4:4

    ((4, 1), (1, 1), (1, 1)),  # 4:1:1
    ((4, 1), (2, 1), (1, 1)),   # Cb 4:2:2 Cr 4:1:1
    ((4, 1), (1, 1), (2, 1)),   # Cb 4:1:1 Cr 4:2:2

    ((4, 2), (1, 1), (1, 1)),  # 4:1:0

    ((1, 4), (1, 1), (1, 1)),  # 1:0.5:0
    ((1, 4), (1, 2), (1, 1)),

    ((2, 4), (1, 1), (1, 1)),  # 2:0.5:0

    ((3, 1), (1, 1), (1, 1)),  # 3:1:1
    ((3, 1), (3, 1), (1, 1)),   # Cb 4:4:4 Cr 3:1:1
    ((3, 1), (1, 1), (3, 1)),   # Cb 3:1:1 Cr 4:4:4
    ((3, 2), (3, 1), (1, 1)),  # 3:3:0
    ((3, 2), (1, 2), (1, 2)),  # 3:1:1
]

implementations = {
    'PIL': implementation.PIL_IO,
    'cv2': implementation.cv2_IO,
    'plt': implementation.plt_IO,
    '6b': implementation.libjpeg6b_IO,
    '8d': implementation.libjpeg8d_IO,
    '9d': implementation.libjpeg9d_IO,
    '9e': implementation.libjpeg9e_IO,
    'turbo': implementation.libjpegturbo_IO
}

cspaces = {1: 'JCS_GRAYSCALE', 3: 'JCS_RGB'}
_flags = {True: ['+DO_FANCY_UPSAMPLING'],
          False: ['-DO_FANCY_UPSAMPLING'], None: []}


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
    decompressor = None


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


def read_jpeg(path: str):
    # read DCT
    return jpeglib.read_dct(path)


def compress_image_read_jpeg(x: np.ndarray, ctx: TestContext):
    with tempfile.NamedTemporaryFile() as tmp:
        compress_image(x, tmp.name, ctx)
        return read_jpeg(tmp.name)
