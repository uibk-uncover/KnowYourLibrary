
from src import TestContext
from src import compression
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
    print("is turbo faster in decompression than 6b:", end="")
    p = is_turbo_faster_than_6b(dataset)
    print(" p-value", p.compression, end="\n\n")

    # baseline
    print("--- baseline ---")
    baseline = compression.run_test(dataset, TestContext())
    compression.print_clusters(baseline)
    print()

    # DCT methods
    print("--- DCT methods ---")
    for dct_method in ['JDCT_ISLOW', 'JDCT_FLOAT', 'JDCT_IFAST']:
        ctx = TestContext()
        ctx.dct_method = dct_method
        print("Method:", dct_method)
        dct = compression.run_test(dataset, ctx)
        compression.print_clusters(dct)
    print()
    
    # quality
    print("--- Quality ---")
    for quality in range(101):
        ctx = TestContext()
        ctx.quality = quality
        print("Quality:", quality)
        q = compression.run_test(dataset, ctx)
        compression.add_print_grouped_clusters(dct, q)
    compression.end_print_grouped_clusters()
    print()


if __name__ == "__main__":
    # get datasets
    db_path = Path.home() / 'Datasets'
    image_dimensions = (512, 512)
    sample_size = 10

    alaska = load_alaska_with_extrems(
        db_path / 'ALASKA_v2_TIFF_256_COLOR', sample_size, (256, 256))
    # boss = load_boss_with_extrems(
    #    db_path / 'BOSS_raw' / 'BOSS_from_raw', sample_size, image_dimensions)

    # get datasets
    #alaska = get_color_dataset(10)
    #boss = get_grayscale_dataset(10)

    # compression tests
    run_compression_tests(alaska)
    # run_compression_tests(alaska)
