
import argparse
import os
from pathlib import Path

from knowyourlibrary import TestContext
from knowyourlibrary import compression
from knowyourlibrary import output
from knowyourlibrary import implementation
from knowyourlibrary import psnr
from knowyourlibrary.dataset import *
from knowyourlibrary.simd import *
from knowyourlibrary._defs import samp_factors, implementations


def run_compression_tests(dataset: np.ndarray):
    """Runs compression tests

    # TODO
    #     - PSNR
    # """
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
    ctx = TestContext()
    res = compression.run_test(dataset, ctx)
    compression.print_clusters(res)
    print(end="\n\n")

    def run_dct_compression_test(samp_factor, use_fancy_sampling=None):
        for dct_method in ['JDCT_ISLOW', 'JDCT_FLOAT', 'JDCT_IFAST']:
            ctx = TestContext()
            ctx.dct_method_compression = dct_method
            ctx.samp_factor = samp_factor
            ctx.use_fancy_sampling = use_fancy_sampling
            print("Method:", dct_method)
            res = compression.run_test(dataset, ctx)
            compression.print_clusters(res)

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
    for quality in range(25, 101):
    #for quality in range(0, 101):
        ctx = TestContext()
        ctx.quality = quality
        res = compression.run_test(dataset, ctx)
        compression.add_print_grouped_clusters(res, quality)
    compression.end_print_grouped_clusters()
    print(end="\n\n")

    # sampling factor
    if dataset.shape[3] == 3:
        print("--- Sampling factor ---")
        # fancy vs. simple
        for use_fancy_sampling, method in zip([True, False], ['Fancy downsampling', 'Simple_scaling']):
            print(method)
            # sampling factores
            for samp_factor in samp_factors:
                ctx = TestContext()
                ctx.samp_factor = samp_factor
                ctx.use_fancy_sampling = use_fancy_sampling
                res = compression.run_test(dataset, ctx)
                compression.add_print_grouped_clusters(res, samp_factor)
            compression.end_print_grouped_clusters()
        print(end="\n\n")

    def run_margin_compression_test(offsets, samp_factor=None, use_fancy_sampling=None, mod=8):
        for d in generate_cropped_datasets(dataset, offsets):
            offset = (d.shape[1] % mod, d.shape[2] % mod)
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
                                        ((2, 2), (1, 1), (1, 1)), use_fancy_sampling, mod=16)
            print()
        print()

    # Python implementations
    print("--- Python implementations ---")
    ctx = TestContext()
    ctx.versions = implementations.copy()
    if dataset.shape[3] == 3:
        ctx.compressor = implementation.io_compressor_rgb
    else:
        ctx.compressor = implementation.io_compressor_grayscale
        del ctx.versions['plt'] # problems with plt compression grayscale
    res = compression.run_test(dataset, ctx)
    compression.print_clusters(res)
    print(end="\n\n")
    
    # compression PSNR
    print("--- Score ---")
    ctx = TestContext()
    ctx.versions = ['6b', '7', '9e']
    res = psnr.run_compression_versions_test(dataset, ctx)
    psnr.TeXize_compression(res)
    print(end="\n\n")


# direct execution
if __name__ == "__main__":
    raise NotImplementedError("module not intended to be executed directly, please use run.py")