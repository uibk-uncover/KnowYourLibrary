
import argparse
import os
from pathlib import Path
import sys
sys.path.append('.')
import run_compression as C
import run_decompression as D
from src.dataset import *

PATH_DEFAULT = {
    'alaska':   str(Path.home() / 'Datasets' / 'ALASKA_v2_TIFF_256_COLOR'),
    'boss':     str(Path.home() / 'Datasets' / 'BOSS_tiles')
}

def parse_args():
    # parse arguments
    parser = argparse.ArgumentParser(description='Know Your Library: comparison of libjpeg versions')
    parser.add_argument('mode', type=str, nargs="*", choices=['all','compression','decompression'], default="all", help='which tests to run')
    parser.add_argument('--input', '-i', type=str, help='input paths')
    parser.add_argument('-n', '--number', type=int, help='number of samples')
    args = parser.parse_args()
    # get paths
    datasets = {}
    if args.input is not None:
        for inarg in args.input.strip().split(';'):
            dataset,*path = inarg.strip().split('=')
            dataset = dataset.strip().lower()
            path = '='.join(path)
            assert dataset in {'alaska','boss'}, 'supported datasets are alaska and boss'
            if not path:
                path = PATH_DEFAULT[dataset]
            assert os.path.isdir(path), 'invalid dataset location'
            datasets[dataset] = path
    else:
        datasets = PATH_DEFAULT
    return {
        'mode': args.mode,
        'path': datasets,
        'number': args.number if args.number else 1000,
    }


def run_all_tests(dataset: np.ndarray):
    """Runs compression and decompression tests.
    
    Args:
        dataset (np.ndarray): Tensor with N images of shape HxWxchannels, shape (N,H,W,channels).
    """
    C.run_compression_tests(dataset)
    D.run_decompression_tests(dataset)

MODE_FUNCTION = {
    'compression':      C.run_compression_tests,
    'decompression':    D.run_decompression_tests,
    'all':              run_all_tests,
}

if __name__ == "__main__":

    # parse arguments
    conf = parse_args()

    # run alaska
    if 'alaska' in conf['path']:
        alaska = load_alaska_with_extremes(
            db_path         = conf['path']['alaska'],
            sample_size     = conf['number'],
            img_dimensions  = (256, 256))
        MODE_FUNCTION[conf['mode']](alaska)
    
    # run boss
    if 'boss' in conf['path']:
        boss = load_boss_with_extremes(
            db_path         = conf['path']['boss'],
            sample_size     = conf['number'],
            img_dimensions  = (512, 512))
        MODE_FUNCTION[conf['mode']](boss)
