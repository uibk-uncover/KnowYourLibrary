
from pathlib import Path
import sys
sys.path.append('src')
from dataset import *
import simd

def run_compression_tests(dataset: np.ndarray):
    # intro
    print("=== Compression tests ===")
    print("Data:")
    print("| Data size: ", dataset.shape[0])
    print("| Image size: ", dataset.shape[1:3])
    print("| Channels: ", dataset.shape[3], end="\n\n")
    
    # performance test: turbo << 6b
    p = simd.is_turbo_faster_than_6b(dataset)
    print("is turbo faster in decompression than 6b: p-value", p.compression)
    
    

if __name__ == "__main__":
    # x = get_checkerboard((16,16),(5,5))
    # import matplotlib.pyplot as plt
    # plt.imshow(x)
    # plt.show()

    alaska = get_color_dataset(100)
    boss = get_grayscale_dataset(100)

    # compression tests
    run_compression_tests(alaska)
    run_compression_tests(alaska)