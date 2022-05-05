
import imp
from src import TestContext
from src import compression
from src import output
from src import implementation
from src.simd import *
from src.dataset import *
from src._defs import samp_factors, implementations
from pathlib import Path
import sys
sys.path.append('.')


def run_compression_tests(dataset: np.ndarray):
    """Runs compression tests

    TODO
        - PSNR
    """
    # intro
    print("=== Compression tests ===", end="\n\n")
    print("--- Data ---")
    output.print_intro(dataset)

    # performance test: turbo << 6b
    print("is turbo faster in decompression than 6b:", end="")
    p = is_turbo_faster_than_6b(dataset)
    print(" p-value", p.compression, end="\n\n")

    # baseline
    print("--- baseline ---")
    res = compression.run_test(dataset, TestContext())
    compression.print_clusters(res)
    print()

    def run_dct_compression_test(samp_factor, use_fancy_sampling=None):
        for dct_method in ['JDCT_ISLOW', 'JDCT_FLOAT', 'JDCT_IFAST']:
            ctx = TestContext()
            ctx.dct_method_compression = dct_method
            ctx.samp_factor = samp_factor
            ctx.use_fancy_sampling = use_fancy_sampling
            print("Method:", dct_method)
            res = compression.run_test(dataset, ctx)
            compression.print_clusters(res)
            del ctx

    # DCT method
    print("--- DCT methods ---")
    print("4:4:4 no downsampling")
    run_dct_compression_test(((1, 1), (1, 1), (1, 1)))
    if dataset.shape[3] == 3:
        for use_fancy_sampling, method in zip([True, False], ['fancy downsampling', 'simple_scaling']):
            print(f"4:2:0 {method}")
            run_dct_compression_test(
                ((2, 2), (1, 1), (1, 1)), use_fancy_sampling)
    print()

    # quality
    print("--- Quality ---")
    for quality in range(0, 101):
        ctx = TestContext()
        ctx.quality = quality
        res = compression.run_test(dataset, ctx)
        compression.add_print_grouped_clusters(res, quality)
        del ctx
    compression.end_print_grouped_clusters()
    print()

    # sampling factor
    if dataset.shape[3] == 3:
        print("--- Sampling factor ---")
        # fancy vs. simple
        for use_fancy_sampling, method in zip([True, False], ['Fancy downsampling', 'Simple_scaling']):
            print(method)
            # sampling factores
            for samp_factor in __defs.samp_factors:
                ctx = TestContext()
                ctx.samp_factor = samp_factor
                ctx.use_fancy_sampling = use_fancy_sampling
                res = compression.run_test(dataset, ctx)
                compression.add_print_grouped_clusters(res, samp_factor)
                del ctx
            compression.end_print_grouped_clusters()
        print()

    def run_margin_compression_test(offsets, samp_factor=None, use_fancy_sampling=None):
        for d in generate_cropped_datasets(dataset, offsets):
            offset = (d.shape[1] % 8, d.shape[2] % 8)
            ctx = TestContext()
            ctx.samp_factor = samp_factor
            ctx.use_fancy_sampling = use_fancy_sampling
            res = compression.run_test(dataset, ctx)
            compression.add_print_grouped_clusters(res, offset)
            del ctx
        compression.end_print_grouped_clusters()

    # margin effects
    print("--- Margin effects ---")
    print("4:4:4 no downsampling")
    run_margin_compression_test([0, 1, 2, 4, 7, 8],
                                ((1, 1), (1, 1), (1, 1)))
    if dataset.shape[3] == 3:
        for use_fancy_sampling, method in zip([True, False], ['fancy downsampling', 'simple_scaling']):
            print(f"4:2:0 {method}")
            run_margin_compression_test([16, 15, 9, 8, 7, 3, 2, 1],
                                        ((2, 2), (1, 1), (1, 1)), use_fancy_sampling)
        print()

    # Python implementations
    print("--- Python implementations ---")
    ctx = TestContext()
    ctx.versions = implementations
    if dataset.shape[3] == 3:
        ctx.compressor = implementation.io_compressor_rgb
    else:
        ctx.compressor = implementation.io_compressor_grayscale
    res = compression.run_test(dataset, ctx)
    compression.print_clusters(res)
    del ctx


if __name__ == "__main__":

    db_path = Path.home() / 'Datasets'
    image_dimensions = (512, 512)
    sample_size = 10

    alaska = load_alaska_with_extrems(
        db_path / 'ALASKA_v2_TIFF_256_COLOR', sample_size, (256, 256))
    boss = load_boss_with_extrems(
        db_path / 'BOSS_raw' / 'BOSS_from_raw', sample_size, image_dimensions)

    # compression tests
    # run_compression_tests(alaska)
    # run_compression_tests(boss)
    # decompression tests
    run_decompression_tests(boss)
    # run_decompression_tests(boss)
