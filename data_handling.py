import matplotlib.pyplot as plt
import os
from pathlib import Path
import random
from typing import Literal, Tuple
import numpy as np


def load_alaska(db_path: Path, sample_size: int) -> list[Path]:
    alaska_path = db_path / 'ALASKA_v2_TIFF_256_COLOR'
    alaska_names = [alaska_path / f for f in os.listdir(alaska_path)]
    print("Loaded ALASKA2 database with", len(alaska_names), "images.")

    random.seed(13245)
    return random.sample(alaska_names, sample_size)


def load_boss(db_path: Path, sample_size: int) -> list[Path]:
    boss_path = db_path / 'BOSS_raw' / 'BOSS_from_raw'  # / 'cover'
    boss_names = [boss_path / f for f in os.listdir(boss_path)]
    print("Loaded BOSS database with", len(boss_names), "images.")

    random.seed(13245)
    return random.sample(boss_names, sample_size)


def get_extrem_saturated_samples(db_names: list[Path]) -> Tuple[Path, Literal]:
    most_saturated, least_saturated = (None, 0), (None, 0)

    for i, f in enumerate(db_names):
        if i % 500 == 0:
            mname = (i + '/' + len(db_names) + ' ' + end='\r')
            print(i, '/', len(db_names), '         ', end='\r')
        if str(f).split('.')[-1] != 'tif':
            continue
        x = plt.imread(str(f))
        xmin, xmax = (x == 0).sum(), (x == 255).sum()
        if xmin > least_saturated[1]:
            least_saturated = (f, xmin)
        if xmax > most_saturated[1]:
            most_saturated = (f, xmax)
    most_saturated, least_saturated = (db_names / '10343.tif',
                                       98491), (db_names / '05887.tif', 78128)


def get_checkerboard(boardsize: int, tilesize: int, channels: int = 3) -> np.uint8:
    board = np.zeros([*boardsize, channels], dtype=np.uint8)
    for i in range(boardsize[0]):
        for j in range(boardsize[1]):
            if (i//tilesize[0]) % 2 == (j//tilesize[1]) % 2:
                board[i, j] = 255
    return board
