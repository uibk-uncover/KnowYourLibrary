
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import random
from typing import Literal, Tuple, List, Union

def load_alaska(size: int = 1000, path: Union[Path,str] = Path.home() / 'Datasets') -> List[str]:
    """"""
    # get alaska
    alaska_path = Path(path) / 'ALASKA_v2_TIFF_256_COLOR'
    alaska_names = [alaska_path / f for f in os.listdir(alaska_path)]
    logging.info(f"found ALASKA2 database with {len(alaska_names)} images")
    # subset
    random.seed(13245)
    return [str(p) for p in random.sample(alaska_names, size)]

def load_boss(size: int = 1000, path: Union[Path,str] = Path.home() / 'Datasets') -> List[str]:
    """"""
    # get bossbase
    boss_path = Path(path) / 'BOSS_raw' / 'BOSS_from_raw' # / 'cover'
    boss_names = [boss_path / f for f in os.listdir(boss_path)]
    logging.info(f"found BOSS database with {len(boss_names)} images")
    # subset
    random.seed(13245)
    return [str(p) for p in random.sample(boss_names, size)]

# def get_extreme_saturated_samples(paths: List[Union[Path,str]]) -> Tuple[Tuple[Path,int],Tuple[Path,int]]:
#     """"""
#     most_saturated, least_saturated = (None, 0), (None, 0)

#     for i, f in enumerate(db_names):
#         if i % 500 == 0:
#             mname = (i + '/' + len(db_names) + ' ')
#             print(i, '/', len(db_names), '         ', end='\r')
#         if str(f).split('.')[-1] != 'tif':
#             continue
#         x = plt.imread(str(f))
#         xmin, xmax = (x == 0).sum(), (x == 255).sum()
#         if xmin > least_saturated[1]:
#             least_saturated = (f, xmin)
#         if xmax > most_saturated[1]:
#             most_saturated = (f, xmax)
#     most_saturated = (db_names / '10343.tif', 98491)
#     least_saturated = (db_names / '05887.tif', 78128)
#     return most_saturated,least_saturated

def get_checkerboard(boardsize: Tuple[int,int], tilesize: Tuple[int,int], channels: int = 3) -> np.ndarray:
    """Constructs checkerboard with three channels of dimensions boardsize. Each tile has dimension tilesize."""
    board = np.zeros([*boardsize, channels], dtype=np.uint8)
    for i in range(boardsize[0]):
        for j in range(boardsize[1]):
            if (i//tilesize[0]) % 2 == (j//tilesize[1]) % 2:
                board[i, j] = 255
    return board

def get_checkerboards(boardsize: Tuple[int,int], channels: int = 3) -> List[np.uint8]:
    """"""
    return np.array([
        get_checkerboard(boardsize, tilesize, channels=channels)
        for tilesize in [(4,4),(7,7),(8,8),(15,15),(16,16)]
    ])

def get_color_dataset(size: int = 1000, path: Union[Path,str] = Path.home() / 'Datasets') -> np.ndarray:
    """Returns color dataset of required size."""
    assert(size >= 5)
    # alaska
    alaska_names = load_alaska(size=size - 7, path=path)
    alaska_path = path / 'ALASKA_v2_TIFF_256_COLOR'
    # special samples
    checkerboard = get_checkerboards((256,256), 3)
    most_saturated = (alaska_path / '10343.tif', 98491)
    least_saturated = (alaska_path / '05887.tif', 78128)
    # load
    alaska = np.array([
        plt.imread(f)
        for f in [*alaska_names, most_saturated[0], least_saturated[0]]
    ])
    return np.concatenate([alaska, checkerboard], axis=0)
    

def get_grayscale_dataset(size: int = 1000, path: Union[Path,str] = Path.home() / 'Datasets') -> np.ndarray:
    """Returns grayscale dataset of required size."""
    assert(size >= 5)
    # alaska
    boss_names = load_boss(size=size - 7, path=path)
    boss_path = path / 'BOSS_raw' / 'BOSS_from_raw'
    # special samples
    checkerboard = get_checkerboards((512,512), 1)
    most_saturated = (boss_path / '6900_1_3.png',262144)
    least_saturated = (boss_path / '6155_1_6.png', 88944)
    # load
    boss = np.array([
        cv2.imread(str(f),cv2.IMREAD_GRAYSCALE)
        for f in [*boss_names, most_saturated[0], least_saturated[0]]
    ])
    boss = np.expand_dims(boss, axis=3)
    return np.concatenate([boss, checkerboard], axis=0)
    