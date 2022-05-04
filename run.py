
from src.simd import *
from src.dataset import *
from pathlib import Path
import sys
sys.path.append('.')


def run_compression_tests(dataset: np.ndarray):
    # intro
    print("=== Compression tests ===")
    print("Data:")
    print("| Data size: ", dataset.shape[0])
    print("| Image size: ", dataset.shape[1:3])
    print("| Channels: ", dataset.shape[3], end="\n\n")

    # performance test: turbo << 6b
    p = is_turbo_faster_than_6b(dataset)
    print("is turbo faster in decompression than 6b: p-value", p.compression)

    # baseline


if __name__ == "__main__":
    # get datasets
    db_path = Path.home() / 'Datasets'
    image_dimensions = Tuple(512, 512)
    sample_size = 10

    alaska = load_alaska_with_extrems(
        db_path / 'ALASKA_v2_TIFF_256_COLOR', sample_size, Tuple(256, 256))
    boss = load_boss_with_extrems(
        db_path / 'BOSS_raw' / 'BOSS_from_raw', sample_size, image_dimensions)

    # compression tests
    run_compression_tests(alaska)
    # run_compression_tests(alaska)
