
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import random
from typing import Tuple, List


def get_dataset(db_path: Path, sample_size: int) -> List[str]:
    # TODO-optimization: generate random sample_size numbers and load at index
    db_names = [db_path / file for file in os.listdir(db_path)]
    print('Loaded', len(db_names), 'images from', str(db_path))
    random.seed(13245)
    print()
    return random.sample(db_names, min(len(db_names),sample_size))


def load_alaska_with_extremes(db_path: Path, sample_size: int, img_dimensions: Tuple[int, int]) -> np.ndarray:
    db_path = Path(db_path)
    db_names = get_dataset(db_path, sample_size)
    checkerboard = get_checkerboards(img_dimensions, 3)
    # TODO: write function
    most_saturated = (db_path / '10343.tif', 98491)
    least_saturated = (db_path / '05887.tif', 78128)
        
    db = np.array([
        plt.imread(file)
        for file in [*db_names, most_saturated[0], least_saturated[0]]
        if file.exists()
    ])

    return np.concatenate([db, checkerboard], axis=0)


def load_boss_with_extremes(db_path: Path, sample_size: int, img_dimensions: Tuple[int, int]) -> np.ndarray:
    db_path = Path(db_path)
    db_names = get_dataset(db_path, sample_size)
    checkerboard = get_checkerboards(img_dimensions, 1)
    most_saturated = (db_path / '6900_1_3.png', 262144)
    least_saturated = (db_path / '6155_1_6.png', 88944)
    
    db = np.array([
        cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        for file in [*db_names, most_saturated[0], least_saturated[0]]
    ], dtype=np.uint8)
    db = np.expand_dims(db, axis=3)
    return np.concatenate([db, checkerboard], axis=0)

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


def get_checkerboard(boardsize: Tuple[int, int], tilesize: Tuple[int, int], channels: int = 3) -> np.ndarray:
    """Constructs checkerboard with three channels of dimensions boardsize. Each tile has dimension tilesize."""
    board = np.zeros([*boardsize, channels], dtype=np.uint8)
    for i in range(boardsize[0]):
        for j in range(boardsize[1]):
            if (i//tilesize[0]) % 2 == (j//tilesize[1]) % 2:
                board[i, j] = 255
    return board


def get_checkerboards(boardsize: Tuple[int, int], channels: int = 3) -> List[np.uint8]:
    """"""
    return np.array([
        get_checkerboard(boardsize, tilesize, channels=channels)
        for tilesize in [(4, 4), (7, 7), (8, 8), (15, 15), (16, 16)]
    ])

def generate_cropped_datasets(dataset: np.ndarray, offsets: list, base: tuple = (128,128)):
    # dimensions
    H,V = dataset.shape[1:3]
    center_h,center_v = int(H/2),int(V/2)
    base_h,base_v = base
    # 2D offsets
    for offset_h in offsets:
        for offset_v in offsets:
            min_h = center_h - int(base_h/2) - offset_h
            min_v = center_v - int(base_v/2) - offset_v
            max_h = center_h + int(base_h/2)
            max_v = center_v + int(base_v/2)
            # crop
            yield dataset.view()[:,min_h:max_h,min_v:max_v,:]

