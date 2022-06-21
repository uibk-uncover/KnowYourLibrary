# KnowYourLibrary

This is the code repository for the paper *Know Your Library: How the libjpeg Version Influences Compression and Decompression Results* [[1](#1)].  It contains the codebase to replicate the systematic comparison of all libjpeg versions between 1998 and 2022.


## Setup

Clone the repository to your computer

```bash
git clone https://github.com/uibk-uncover/KnowYourLibrary
cd KnowYourLibrary
```

Install the Python dependencies in a clean virtual environment.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> :exclamation: Note that the installation takes a few minutes.  This is because the libjpeg versions (part of package [jpeglib](https://pypi.org/project/jpeglib/)) are source distributions to be compiled on your computer.

## Usage

The test cases are executed with the following command.

```bash
python run.py
```

The script accepts the argument `--help` for usage instructions and availaible options.


```bash
python run.py
    [all|compression|decompression]
    [-i|--input "alaska=<path-to-alaska>;boss=<path-to-boss>"]
    [-n|--number <number-of-samples>]
    [--help|-h]
```

Using this interface, you can specify whether to test compression, decompression or both.  You can also specify whether to run for colored (using the ALASKA dataset [[2]](#2)) or grayscale images (using the BOSSBase dataset [[3]](#3)).  You can overwrite the default location with a custom one.  In addition you can choose how many images from dataset will be used.


By default program uses at most 1000 images + certain specifically chosen (with maximal and minimal saturation, synthetic "checkerboard" with sharp edges etc.).  The default location of the ALASKA dataset is `~/Datasets/ALASKA_v2_TIFF_256_COLOR`, for BOSSBase it is `~/Datasets/BOSS_tiles`.

### Examples

To run compression test on ALASKA in directory `/alaska` and BOSSBase in default directory, type

```bash
python run.py compression -i "alaska=/alaska;boss"
```

To run decompression test on BOSSBase in directory `/data/boss`, but only use 30 images, type

```bash
python run.py decompression -i "boss=/data/boss" -n 30
```

There are example images in this repository. Execute tests on them only with

```bash
python run.py all -i "alaska=./data/alaska;boss=./data/boss" -n 15
```



## Repository structure

The following files and directories contain the experiments.

- `run.py` = entrypoint for executing
- `run_compression.py` = structure of compression tests
- `run_decompression.py` = structure of decompression tests
- `src/` = Python implementation
- `data/*.sha256` = SHA256 hashes of the files

The following files are intended to facilitate the repuducibility.

- `requirements.txt` = Python dependencies
- `Dockerfile` = Docker file
- `data/alaska`, `data/boss` = 15 example files from each dataset, to see limited results without need to get full datasets

## References

<a id="1">[1]</a>
M. Benes, N. Hofer, and R. Böhme. 2022. Know Your Library:
How the libjpeg Version Influences Compression and Decompression Results. In IH&MMSec. ACM, ?-?.

<a id="2">[2]</a> 
R. Cogranne, Q. Giboulot, and P. Bas. 2019. The ALASKA steganalysis challenge:
A first step towards steganalysis. In IH&MMSec. ACM, 125–137.

<a id="3">[3]</a> 
P. Bas, T. Filler, and T. Pevný. 2011. Break our steganographic system. In IH
(LNCS, Vol. 6958). Springer, 59–70.
