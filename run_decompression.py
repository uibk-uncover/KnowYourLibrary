
from src import psnr
from src import implementation, psnr
from src import decompression
from src import TestContext
from src import output
from src import dataset
from src.dataset import *
from src._defs import samp_factors, implementations

from pathlib import Path
import sys
sys.path.append('.')


def run_baseline(dataset: np.ndarray):
    ctx = TestContext()
    baseline = decompression.run_test(dataset, ctx)
    output.print_clusters(baseline)


def run_dct(dataset: np.ndarray, samp_factor: Tuple[int], use_fancy_sampling=None):
    ctx = TestContext()
    ctx.samp_factor = samp_factor
    ctx.use_fancy_sampling = use_fancy_sampling

    for dct_method in ['JDCT_ISLOW', 'JDCT_FLOAT', 'JDCT_IFAST']:
        ctx.dct_method_decompression = dct_method
        print("Method:", ctx.dct_method_decompression)
        dct_result = decompression.run_test(dataset, ctx)
        output.print_clusters(dct_result)


def run_dct_with_sampling(dataset: np.ndarray):
    for use_fancy_sampling, method in zip([True, False], ['fancy upsampling', 'simple scaling']):
        print(f"4:2:0 {method}")
        run_dct(dataset,
                ((2, 2), (1, 1), (1, 1)), use_fancy_sampling)


def run_quality(dataset: np.ndarray):
    ctx = TestContext()

    for quality in range(25, 101):
        ctx.quality = quality
        quality_result = decompression.run_test(dataset, ctx)
        output.add_print_grouped_clusters(quality_result.spatial, quality)
    output.end_print_grouped_clusters()
    print()


def run_sampling_factor(dataset: np.ndarray):
    ctx = TestContext()

    for use_fancy_sampling, method in zip([True, False], ['Fancy upsampling', 'Simple scaling']):
        print(method)
        ctx.use_fancy_sampling = use_fancy_sampling

        for samp_factor in samp_factors:
            ctx.samp_factor = samp_factor
            ctx.use_fancy_sampling = use_fancy_sampling
            sampling_factor_result = decompression.run_test(dataset, ctx)

            output.add_print_grouped_clusters(
                sampling_factor_result.spatial, samp_factor)
        output.end_print_grouped_clusters()


def run_margin_effects(dataset: np.ndarray, offsets: list[int], samp_factor=None, use_fancy_sampling=None):
    ctx = TestContext()
    ctx.samp_factor = samp_factor
    ctx.use_fancy_sampling = use_fancy_sampling

    for data in generate_cropped_datasets(dataset, offsets):
        offset = (data.shape[1] % 8, data.shape[2] % 8)
        margin_result = decompression.run_test(data, ctx)
        output.add_print_grouped_clusters(margin_result.spatial, offset)
    output.end_print_grouped_clusters()


def run_margin_with_sampling(dataset: np.ndarray, offsets: list[int]):
    for use_fancy_sampling, method in zip([True, False], ['fancy upsampling', 'simple scaling']):
        print(f"4:2:0 {method}")
        run_margin_effects(dataset, offsets, use_fancy_sampling)


def run_python_implementation(dataset: np.ndarray):
    ctx = TestContext()
    ctx.versions = implementations
    ctx.decompressor = implementation.io_decompressor

    implementation_results = decompression.run_test(dataset, ctx)
    output.print_clusters(implementation_results)


def run_PSNR_dct(dataset: np.ndarray, dct_method_v1: str, dct_method_v2: str):
    ctx_1 = TestContext()
    ctx_1.dct_method_decompression = dct_method_v1
    ctx_2 = TestContext()
    ctx_2.dct_method_decompression = dct_method_v2

    psnr.print_PSNR(dataset, TestContext(), ctx_1, ctx_2, 'DCT')


def run_PSNR_qf(dataset: np.ndarray, qf_1: int, qf_2: int):
    ctx_1 = TestContext()
    ctx_1.quality = qf_1
    ctx_2 = TestContext()
    ctx_2.quality = qf_2

    print('qf1 vs qf2 : ', qf_1, qf_2)
    psnr.print_PSNR(dataset, TestContext(), ctx_1, ctx_2, 'qf')


def run_PSNR_version(dataset: np.ndarray, v1: str, v2: str):
    ctx_1 = TestContext()
    ctx_1.v_arbitrary = v1
    ctx_2 = TestContext()
    ctx_2.v_arbitrary = v2

    psnr.print_PSNR(dataset, TestContext(), ctx_1, ctx_2, 'version')


def run_decompression_tests(dataset: np.ndarray):

    data_is_color = (dataset.shape[3] == 3)

    print("=== Decompression tests ===")
    output.print_intro(dataset)

    print("--- baseline ---")
    run_baseline(dataset)

    print("--- DCT methods ---")
    print("4:4:4 no upsampling")
    run_dct(dataset, ((1, 1), (1, 1), (1, 1)))
    if data_is_color:
        run_dct_with_sampling(dataset)

    print("--- Quality ---")
    run_quality(dataset)
    if data_is_color:
        print("--- Sampling factor ---")
        run_sampling_factor(dataset)

    print("--- Margin effects ---")
    print("4:4:4 no downsampling")
    run_margin_effects(dataset, [0, 1, 2, 4, 7, 8],
                       ((1, 1), (1, 1), (1, 1)))
    if data_is_color:
        run_margin_with_sampling(dataset, [16, 15, 9, 8, 7, 3, 2, 1],
                                 ((2, 2), (1, 1), (1, 1)))

    print("--- Python implementations ---")
    run_python_implementation(dataset)

    run_PSNR_dct(alaska, 'JDCT_ISLOW', 'JDCT_IFAST')
    run_PSNR_dct(alaska, 'JDCT_ISLOW', 'JDCT_FLOAT')

    print('------- GRAYSCALE ----------')
    run_PSNR_dct(boss, 'JDCT_ISLOW', 'JDCT_IFAST')
    run_PSNR_dct(boss, 'JDCT_ISLOW', 'JDCT_FLOAT')

    print('---------VERSIONS----------')
    run_PSNR_version(alaska, 'turbo', '9')
    run_PSNR_version(alaska, '7', '9a')
    run_PSNR_version(alaska, '6b', '9a')

    print('------- QUALITY ----------')
    run_PSNR_qf(alaska, 75, 90)
    run_PSNR_qf(alaska, 90, 95)
    run_PSNR_qf(alaska, 95, 100)


if __name__ == "__main__":

    db_path = Path.home() / 'Datasets'
    image_dimensions = (512, 512)
    sample_size = 993

    alaska = load_alaska_with_extrems(
        db_path / 'ALASKA_v2_TIFF_256_COLOR', sample_size, (256, 256))
    boss = load_boss_with_extrems(
        db_path / 'BOSS_raw' / 'BOSS_from_raw', sample_size, image_dimensions)

    print('Running decompression tests ...')
    # run_decompression_tests(alaska)
    # run_decompression_tests(boss)

    run_PSNR_dct(alaska, 'JDCT_ISLOW', 'JDCT_IFAST')
    run_PSNR_dct(alaska, 'JDCT_ISLOW', 'JDCT_FLOAT')

    print('------- GRAYSCALE ----------')
    run_PSNR_dct(boss, 'JDCT_ISLOW', 'JDCT_IFAST')
    run_PSNR_dct(boss, 'JDCT_ISLOW', 'JDCT_FLOAT')

    print('---------VERSIONS----------')
    run_PSNR_version(alaska, 'turbo', '9')
    run_PSNR_version(alaska, '7', '9a')
    run_PSNR_version(alaska, '6b', '9a')

    print('------- QUALITY ----------')
    run_PSNR_qf(alaska, 75, 90)
    run_PSNR_qf(alaska, 90, 95)
    run_PSNR_qf(alaska, 95, 100)

    # run_python_implementation(alaska)
    # run_python_implementation(boss)
