
from pathlib import Path

from knowyourlibrary import psnr
from knowyourlibrary import implementation, psnr
from knowyourlibrary import decompression
from knowyourlibrary import TestContext
from knowyourlibrary import output
from knowyourlibrary import dataset
from knowyourlibrary.dataset import *
from knowyourlibrary._defs import samp_factors, implementations


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


def run_margin_effects(dataset: np.ndarray, offsets: List[int], samp_factor=None, use_fancy_sampling=None, mod=8):
    for data in generate_cropped_datasets(dataset, offsets):
        offset = (data.shape[1] % mod, data.shape[2] % mod)
        
        ctx = TestContext()
        ctx.samp_factor = samp_factor
        ctx.use_fancy_sampling = use_fancy_sampling
        
        margin_result = decompression.run_test(data, ctx)
        #print(margin_result, offset)
        output.add_print_grouped_clusters(margin_result.spatial, offset)
    output.end_print_grouped_clusters()


def run_margin_with_sampling(dataset: np.ndarray, offsets: List[int], mod=8):
    for use_fancy_sampling, method in zip([True, False], ['fancy upsampling', 'simple scaling']):
        print(f"4:2:0 {method}")
        run_margin_effects(dataset, offsets, ((2,2),(1,1),(1,1)), use_fancy_sampling, mod)


def run_python_implementation(dataset: np.ndarray):
    ctx = TestContext()
    ctx.versions = implementations
    if dataset.shape[3] == 3:
        ctx.decompressor = implementation.io_decompressor_rgb
    else:
        ctx.decompressor = implementation.io_decompressor_grayscale
    implementation_results = decompression.run_test(dataset, ctx)
    output.print_clusters(implementation_results)
    print()


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

    print("--- QUALITY ---")
    run_quality(dataset)
    if data_is_color:
        print("--- SAMPLING FACTOR ---")
        run_sampling_factor(dataset)

    print("--- MARGIN EFFECTS ---")
    print("4:4:4 no downsampling")
    run_margin_effects(dataset, [0, 1, 2, 4, 7, 8],
                       ((1, 1), (1, 1), (1, 1)))
    if data_is_color:
        run_margin_with_sampling(dataset, [16, 15, 9, 8, 7, 3, 2, 1], 16)

    print('------- PYTHON IMPLEMENTATIONS ----------')
    run_python_implementation(dataset)

    # decompression PSNR
    print("--- PSNR ---")
    run_PSNR_dct(dataset, 'JDCT_ISLOW', 'JDCT_IFAST')
    run_PSNR_dct(dataset, 'JDCT_ISLOW', 'JDCT_FLOAT')

    print('--------- PSNR: VERSIONS----------')
    run_PSNR_version(dataset, 'turbo', '9')
    run_PSNR_version(dataset, '7', '9a')
    run_PSNR_version(dataset, '6b', '9a')

    print('------- PSNR: QUALITY ----------')
    run_PSNR_qf(dataset, 75, 90)
    run_PSNR_qf(dataset, 90, 95)
    run_PSNR_qf(dataset, 95, 100)


# direct execution
if __name__ == "__main__":
    raise NotImplementedError("module not intended to be executed directly, please use run.py")