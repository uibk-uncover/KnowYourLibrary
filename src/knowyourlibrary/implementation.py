
import cv2
import jpeglib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import List


class ImageIO:
    """Interface for image i/o classes."""

    def __init__(self, name): self.name = name
    def compress_rgb(self, x): raise NotImplementedError
    def compress_grayscale(self, x): raise NotImplementedError
    def decompress_rgb(self): raise NotImplementedError
    def decompress_grayscale(self): raise NotImplementedError


class PIL_IO(ImageIO):
    """i/o implementation using pillow"""

    def compress_rgb(self, x): Image.fromarray(x).save(self.name)

    def compress_grayscale(self, x):
        # correct
        if len(x.shape) == 2:
            im = Image.fromarray(x)
        # expanded channel dimension
        elif x.shape[2] == 1:
            im = Image.fromarray(x[:, :, 0])
        # colored
        else:
            im = Image.fromarray(x, 'L')
        im.save(self.name)

    def decompress_rgb(self): return np.array(Image.open(self.name))
    def decompress_grayscale(self):
        x = np.array(Image.open(self.name))
        x = np.expand_dims(x, 2)
        return x

class cv2_IO(ImageIO):
    """i/o implementation using opencv"""

    def compress_rgb(self, x): cv2.imwrite(
        self.name, cv2.cvtColor(x, cv2.COLOR_BGR2RGB))

    def compress_grayscale(self, x):
        # correct
        if len(x.shape) == 2:
            cv2.imwrite(self.name, x)
        # expanded channel dimension
        elif x.shape[2] == 1:
            cv2.imwrite(self.name, x[:, :, 0])
        # colored
        else:
            cv2.imwrite(self.name, cv2.cvtColor(x, cv2.COLOR_BGR2RGB))

    def decompress_rgb(self):
        return cv2.cvtColor(cv2.imread(self.name), cv2.COLOR_BGR2RGB)

    def decompress_grayscale(self):
        x = cv2.imread(self.name, cv2.IMREAD_GRAYSCALE)
        x = np.expand_dims(x, 2)
        return x

class plt_IO(ImageIO):
    """i/o implementation using matplotlib"""

    def compress_rgb(self, x): plt.imsave(self.name, x)
    def compress_grayscale(self, x): plt.imsave(self.name, x[:, :, 0], cmap='gray')

    def decompress_rgb(self):
        x = plt.imread(self.name)
        return x

    def decompress_grayscale(self):
        x = plt.imread(self.name)
        x = np.expand_dims(x, 2)
        return x

# libjpeg
class libjpeg_IO(ImageIO):
    def __init__(self, name, version):
        super().__init__(name)
        self.version = version

    def compress_rgb(self, x):
        with jpeglib.version(self.version):
            im = jpeglib.from_spatial(x)
            im.write_spatial(self.name)

    def compress_grayscale(self, x):
        with jpeglib.version(self.version):
            im = jpeglib.from_spatial(x)
            im.jpeg_color_space = jpeglib.Colorspace('JCS_GRAYSCALE')
            im.write_spatial(self.name)

    def decompress_rgb(self):
        with jpeglib.version(self.version):
            im = jpeglib.read_spatial(
                self.name, flags=['+DO_FANCY_UPSAMPLING', '+DO_BLOCK_SMOOTHING'])
            return im.spatial
    def decompress_grayscale(self):
        with jpeglib.version(self.version):
            img = jpeglib.read_spatial(
                self.name, out_color_space='JCS_GRAYSCALE',
                flags=['+DO_FANCY_UPSAMPLING', '+DO_BLOCK_SMOOTHING'])
            return img.spatial

class libjpeg6b_IO(libjpeg_IO):
    def __init__(self, name): super().__init__(name, '6b')


class libjpeg8d_IO(libjpeg_IO):
    def __init__(self, name): super().__init__(name, '8d')


class libjpeg9d_IO(libjpeg_IO):
    def __init__(self, name): super().__init__(name, '9d')


class libjpeg9e_IO(libjpeg_IO):
    def __init__(self, name): super().__init__(name, '9e')


class libjpegturbo_IO(libjpeg_IO):
    def __init__(self, name): super().__init__(name, 'turbo210')


def io_compressor_rgb(dataset, fnames, version, ctx):
    # parse
    impl = ctx.versions[version]
    # compress
    [
        impl(fnames[i]).compress_rgb(dataset[i])
        for i, fname in enumerate(fnames)
    ]


def io_compressor_grayscale(dataset, fnames, version, ctx):
    # parse
    impl = ctx.versions[version]
    # compress
    [
        impl(fnames[i]).compress_grayscale(dataset[i])
        for i, fname in enumerate(fnames)
    ]


def io_decompressor_rgb(fnames: List[str], version, ctx) -> np.ndarray:

    impl = ctx.versions[version]
    return np.stack([
        impl(fname).decompress_rgb()
        for fname in fnames
    ], axis=0)

def io_decompressor_grayscale(fnames: List[str], version, ctx) -> np.ndarray:

    impl = ctx.versions[version]
    return np.stack([
        impl(fname).decompress_grayscale()
        for fname in fnames
    ], axis=0)