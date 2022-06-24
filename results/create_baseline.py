
import jpeglib
import numpy as np
import os
from PIL import Image
import tempfile

VERSIONS = ['6b','turbo210','7','8','8a','8b','8c','8d','9','9a','9b','9c','9d','9e']

# compression
for f in os.listdir('data/alaska'):
    basename = '.'.join(f.split('.')[:-1])
    x = np.array(Image.open(f'data/alaska/{basename}.tif'))
    for v in VERSIONS:
        with jpeglib.version(v):
            jpeglib.from_spatial(x).write_spatial(f'results/compression/baseline_color/{basename}_{v}.jpeg')
for f in os.listdir('data/boss'):
    basename = '.'.join(f.split('.')[:-1])
    x = np.array(Image.open(f'data/boss/{basename}.png'))
    x = np.expand_dims(x, axis=2)
    for v in VERSIONS:
        with jpeglib.version(v):
            jpeglib.from_spatial(x).write_spatial(f'results/compression/baseline_grayscale/{basename}_{v}.jpeg')

# decompression
for f in os.listdir('data/alaska'):
    basename = '.'.join(f.split('.')[:-1])
    x = np.array(Image.open(f'data/alaska/{basename}.tif'))
    with tempfile.NamedTemporaryFile(suffix='.jpeg') as tmp:
        with jpeglib.version('9e'):
            jpeglib.from_spatial(x).write_spatial(tmp.name)
        for v in VERSIONS:
            with jpeglib.version(v):
                x = jpeglib.read_spatial(tmp.name).spatial
                Image.fromarray(x).save(f'results/decompression/baseline_color/{basename}_{v}.png')
for f in os.listdir('data/boss'):
    basename = '.'.join(f.split('.')[:-1])
    x = np.array(Image.open(f'data/boss/{basename}.png'))
    x = np.expand_dims(x, axis=2)
    with tempfile.NamedTemporaryFile(suffix='.jpeg') as tmp:
        with jpeglib.version('9e'):
            jpeglib.from_spatial(x).write_spatial(tmp.name)
        for v in VERSIONS:
            with jpeglib.version(v):
                x = jpeglib.read_spatial(tmp.name).spatial
                Image.fromarray(x[:,:,0]).save(f'results/decompression/baseline_grayscale/{basename}_{v}.png')
