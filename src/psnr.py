
import pandas as pd
import tempfile
from ._defs import *


def PSNR(x_ref, x_noisy):
    """PSNR of two signals."""
    D = np.array(x_ref).astype(np.float32) - \
        np.array(x_noisy).astype(np.float32)
    mse = (D**2).mean()
    maxi = 255  # x_ref.max().astype(np.float64)
    return np.log10(maxi**2/mse)*10


def get_mismatching_images_decomp(psnr):
    psrn = np.array(psnr)
    return (~np.isinf(psrn)).sum()


def get_quantile_decomp(psnr):
    psnr = np.array(psnr)
    psnr = psnr[~np.isinf(psnr)]
    median = np.median(psnr)
    q5 = np.quantile(psnr, .05)
    q95 = np.quantile(psnr, .95)
    return median, q5, q95


def DCT_match_nz(dct1, dct2):
    (Y1, Cb1, Cr1), (Y2, Cb2, Cr2) = dct1, dct2
    dY = (Y1[Y1 != 0] == Y2[Y1 != 0]).sum()
    dCb = (Cb1[Cb1 != 0] == Cb2[Cb1 != 0]).sum()
    dCr = (Cr1[Cr1 != 0] == Cr2[Cr1 != 0]).sum()
    # % of matched DCT coefficients (Y + CbCr)
    return (dY + dCb + dCr) / ((Y1 != 0).sum() + (Cb1 != 0).sum() + (Cr1 != 0).sum())


def DCT_mismatch_log10(dct1, dct2):
    (Y1, Cb1, Cr1), (Y2, Cb2, Cr2) = dct1, dct2
    dY = (Y1 != Y2).sum()
    dCb = (Cb1 != Cb2).sum()
    dCr = (Cr1 != Cr2).sum()
    # % of matched DCT coefficients (Y + CbCr)
    match_pct = (dY + dCb + dCr) / (Y1.size + Cb1.size + Cr1.size)
    return np.log10(match_pct)


def get_missing(match):
    match = np.array(match)
    return np.isnan(match)


def get_mismatching(match):
    return (~np.isinf(match))


def get_q05_q5_q95(match):
    """Get quantiles 5%, 50%, 95%."""
    return np.quantile(match, [.05, .5, .95])

# def print_evaluation_comp_version(**comp):
#     print(comp["v1"], " vs. ", comp["v2"])

#     nz_match, log_match = compression_versions(**comp)
#     matches = [nz_match, log_match]

#     print_var = ["NZ", "LOG"]
#     for i, match in enumerate(matches):
#         if match:
#             median, q5, q95 = get_quantile_comp(match)
#         print(print_var[i])
#         print(get_missing_img_comp(match).sum(), "/", alaska.shape[0], "missing images")
#         print(get_mismatching_img_comp(match).sum(), "/", alaska.shape[0], "mismatching images")
#         print("median: ", median, " q5: ", q5, " q95: ", q95 )
#         print(f"{comp['v1']} vs {comp['v2']} & ${get_mismatching_img_comp(match).sum()}$ & ${round(q5, 2)}$ & ${round(median, 2)}$ & ${round(q95, 2)}$ \\")


# def print_evaluation_comp_para(**comp):
#     if "dct1" in comp:
#         print("DCT: ", comp["dct1"], " vs. ", comp["dct2"])
#     if "qf1" in comp:
#         print("QF: ", comp["qf1"], " vs. ", comp["qf2"])
#     if "flag1" in comp:
#         print("FLAG: ",comp["flag1"], " vs. ", comp["flag2"] if comp["flag2"] is not None else "<empty>")

#     nz_match, log_match = compression_parameters(**comp)
#     matches = [nz_match, log_match]

#     print_var = ["NZ", "LOG"]
#     for i, match in enumerate(matches):
#         if match:
#             median, q5, q95 = get_quantile_comp(match)
#         print(print_var[i])
#         print(get_missing_img_comp(match).sum(), "/", alaska.shape[0], "missing images")
#         print(get_mismatching_img_comp(match).sum(), "/", alaska.shape[0], "mismatching images")
#         print(" q5: ", q5, "median: ", median, " q95: ", q95 )
#         print(f"& ${get_mismatching_img_comp(match).sum()}$ & ${round(q5, 2)}$ & ${round(median, 2)}$& ${round(q95, 2)}$ \\")


def run_compression_versions_test(dataset: np.ndarray, ctx: TestContext):
    """Executes comparison test between versions."""
    # parse
    N, _, _, channels = dataset.shape
    ctx.colorspace = cspaces[channels]

    # temporary directory
    tmp = tempfile.TemporaryDirectory()

    # iterate versions
    matches = {'version': [], 'descriptor': [], 'nz': [], 'log': []}
    for i1, v1 in enumerate(ctx.versions):
        for v2 in ctx.versions[i1+1:]:
            # compress with v1
            with jpeglib.version(v1):
                jpeg1 = [
                    compress_image_read_jpeg(dataset[i], ctx)
                    for i in range(N)
                ]
            # compress with v2
            with jpeglib.version(v2):
                jpeg2 = [
                    compress_image_read_jpeg(dataset[i], ctx)
                    for i in range(N)
                ]
            # compute nz match
            for i in range(N):
                matches['version'].append(v1)
                matches['descriptor'].append(v2)
                matches['nz'].append(
                    DCT_match_nz(
                        (jpeg1[i].Y, jpeg1[i].Cb, jpeg1[i].Cr),
                        (jpeg2[i].Y, jpeg2[i].Cb, jpeg2[i].Cr),
                    )
                )
                matches['log'].append(
                    DCT_mismatch_log10(
                        (jpeg1[i].Y, jpeg1[i].Cb, jpeg1[i].Cr),
                        (jpeg2[i].Y, jpeg2[i].Cb, jpeg2[i].Cr),
                    )
                )

    return pd.DataFrame(matches)


def TeXize_compression(res: pd.DataFrame):
    def quantile(n):
        def quantile_(x):
            return np.quantile(x, n)
        q = '%4.2f' % n
        quantile_.__name__ = f'q{q[2:]}'
        return quantile_
    #        print(get_missing_img_comp(match).sum(), "/", alaska.shape[0], "missing images")
    #    print(get_mismatching_img_comp(match).sum(), "/", alaska.shape[0], "mismatching images")
    res = (
        res
        .groupby(['version', 'descriptor'])
        .agg({
            'nz': [quantile(.05), quantile(.5), quantile(.95)],
            'log': [quantile(.05), quantile(.5), quantile(.95)]})
        .reset_index(drop=False)
    )
    print(res)


# def print_evaluation_comp_para(**comp):
#     if "dct1" in comp:
#         print("DCT: ", comp["dct1"], " vs. ", comp["dct2"])
#     if "qf1" in comp:
#         print("QF: ", comp["qf1"], " vs. ", comp["qf2"])
#     if "flag1" in comp:
#         print("FLAG: ",comp["flag1"], " vs. ", comp["flag2"] if comp["flag2"] is not None else "<empty>")

#     nz_match, log_match = compression_parameters(**comp)
#     matches = [nz_match, log_match]

#     print_var = ["NZ", "LOG"]
#     for i, match in enumerate(matches):
#         if match:
#             median, q5, q95 = get_quantile_comp(match)
#         print(print_var[i])
#         print(get_missing_img_comp(match).sum(), "/", alaska.shape[0], "missing images")
#         print(get_mismatching_img_comp(match).sum(), "/", alaska.shape[0], "mismatching images")
#         print(" q5: ", q5, "median: ", median, " q95: ", q95 )
#         print(f"& ${get_mismatching_img_comp(match).sum()}$ & ${round(q5, 2)}$ & ${round(median, 2)}$& ${round(q95, 2)}$ \\")


def return_PSNR(dataset: np.ndarray, ctx_arb: TestContext,  ctx_1: TestContext, ctx_2: TestContext):
    tmp = tempfile.NamedTemporaryFile()  # create temporary file
    psnr = []
    for i in range(dataset.shape[0]):
        # compress with arbitrary
        with jpeglib.version(ctx_arb.v_arbitrary):
            compress_image(dataset[i], tmp.name, ctx_arb)

        # decompress with each version
        with jpeglib.version(ctx_1.v_arbitrary):
            x_v1 = decompress_image(tmp.name, ctx_1)

        with jpeglib.version(ctx_2.v_arbitrary):
            x_v2 = decompress_image(tmp.name, ctx_2)

        # compute psnr
        psnr.append(PSNR(x_v1.spatial, x_v2.spatial))
    return psnr


def print_PSNR(dataset: np.ndarray, ctx_arb: TestContext, ctx_1: TestContext, ctx_2: TestContext, PSNR_test: str):
    if(PSNR_test == 'version'):
        print("version: ", ctx_1.v_arbitrary,
              " vs. ", ctx_2.v_arbitrary)

    if(PSNR_test == 'DCT'):
        print("DCT: ", ctx_1.dct_method_decompression,
              " vs. ", ctx_2.dct_method_decompression)

    if(PSNR_test == 'qf'):
        print("quality: ", ctx_1.quality,
              " vs. ", ctx_2.quality)

    psnr = return_PSNR(dataset, ctx_arb, ctx_1, ctx_2)

    print(get_mismatching_images_decomp(psnr), "/",
          dataset.shape[0], "mismatching images")
    if psnr:
        print(psnr)
        median, q5, q95 = get_quantile_decomp(psnr)
    print(" q5: ", q5, "median: ", median, " q95: ", q95)
    print(f"& ${get_mismatching_images_decomp(psnr)}$ & ${round(q5, 2)}$ & ${round(median, 2)}$ & ${round(q95, 2)}$ \\")
