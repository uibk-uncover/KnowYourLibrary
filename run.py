
from pathlib import Path
import sys
sys.path.append('.')
from src.dataset import *
from src.simd import *

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
    alaska = get_color_dataset(100)
    boss = get_grayscale_dataset(100)

    # compression tests
    run_compression_tests(alaska)
    #run_compression_tests(alaska)