
from src import TestContext
from src import compression
from src import decompression
from src import output
from src.decompression import run_test
from src.simd import *
from src.dataset import *
from pathlib import Path
import sys
sys.path.append('.')


def run_compression_tests(dataset: np.ndarray):
    # intro
    print("=== Compression tests ===")
    output.print_intro(dataset)

    # performance test: turbo << 6b
    print("is turbo faster in decompression than 6b:", end="")
    p = is_turbo_faster_than_6b(dataset)
    print(" p-value", p.compression, end="\n\n")

    # baseline
    print("--- baseline ---")
    baseline = compression.run_test(dataset, TestContext())
    output.print_clusters(baseline)
    print()

    # DCT methods
    print("--- DCT methods ---")
    for dct_method in ['JDCT_ISLOW', 'JDCT_FLOAT', 'JDCT_IFAST']:
        ctx = TestContext()
        ctx.dct_method_compression = dct_method
        print("Method:", ctx.dct_method_compression)
        dct = compression.run_test(dataset, ctx)
        output.print_clusters(dct)
    print()


def run_decompression_tests(dataset: np.ndarray):
    # intro
    print("=== Decompression tests ===")
    output.print_intro(dataset)

   # baseline
    print("--- baseline ---")
    baseline = decompression.run_test(dataset, TestContext())
    output.print_clusters(baseline)
    print()

    # DCT methods
    print("--- DCT methods ---")
    for dct_method in ['JDCT_ISLOW', 'JDCT_FLOAT', 'JDCT_IFAST']:
        ctx = TestContext()
        ctx.dct_method_decompression = dct_method
        print("Method:", ctx.dct_method_decompression)
        dct = decompression.run_test(dataset, ctx)
        output.print_clusters(dct)
    print()


if __name__ == "__main__":

    db_path = Path.home() / 'Datasets'
    image_dimensions = (512, 512)
    sample_size = 10

    alaska = load_alaska_with_extrems(
        db_path / 'ALASKA_v2_TIFF_256_COLOR', sample_size, (256, 256))
    # boss = load_boss_with_extrems(
    #    db_path / 'BOSS_raw' / 'BOSS_from_raw', sample_size, image_dimensions)

    run_compression_tests(alaska)
    run_decompression_tests(alaska)
