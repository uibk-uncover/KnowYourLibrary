
# for PNGs later
import numpy as np
from PIL import Image

# load base lena
import jpeglib
im = jpeglib.read_spatial("lena.jpeg", out_color_space='JCS_GRAYSCALE')
# get spatial
x = im.spatial

# read image with version and dct method
def write_spatial(x, version, dct_method):
    # set version
    with jpeglib.version(version):
        # write image with dct method
        im = jpeglib.from_spatial(x)
        im.jpeg_color_space = jpeglib.Colorspace("JCS_GRAYSCALE")
        im.dct_method = dct_method
        im.write_spatial(f"lena_{version}_{dct_method}.jpeg", dct_method=dct_method)
        # load jpeg again
        return np.array(Image.open(f"lena_{version}_{dct_method}.jpeg"))

# write image 6 times (6b and 9b, all DCT methods)
x_6b_slow = write_spatial(x, "6b", "JDCT_ISLOW")
x_6b_fast = write_spatial(x, "6b", "JDCT_IFAST")
x_6b_float = write_spatial(x, "6b", "JDCT_FLOAT")
x_9b_slow = write_spatial(x, "9b", "JDCT_ISLOW")
x_9b_fast = write_spatial(x, "9b", "JDCT_IFAST")
x_9b_float = write_spatial(x, "9b", "JDCT_FLOAT")

# check equality
print("islow:", (x_6b_slow == x_9b_slow).all())
print("ifast:", (x_6b_fast == x_9b_fast).all())
print("float:", (x_6b_float == x_9b_float).all())