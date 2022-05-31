# KnowYourLibrary


Systematic comparison of all libjepg versions since 1998 as described in "Know Your Library: How the libjpeg Version Influences Compression and Decompression Results", IH&MMSec 2022.



## Requirements

To run this demonstration, create a virtual environment, source it and install the required Python packages:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Note that this installes the jpeglib library by [Martin Beneš](https://github.com/martinbenes1996), which might take some time.


## Running the comparison
The script takes a subsample of 15 color images from the ALASKA2 dataset[[1]](#1) and 15 grayscale images from the BOSSBase dataset[[1]](#1).
The comparison is available for compression and decompression with given parameters using given versions fo the libjepg library.

The resulting decompression output images are compared in the spatial domain with differences amplified by setting the respective RGB values to a maximum intensity of 255.


Usage:
```
python run.py
    [--compression]
    [--decompression]
    [--*]
```




## References

<a id="1">[1]</a> 
R. Cogranne, Q. Giboulot, and P. Bas. 2019. The ALASKA steganalysis challenge:
A first step towards steganalysis. In IH&MMSec. ACM, 125–137.

<a id="2">[2]</a> 
P. Bas, T. Filler, and T. Pevn `y. 2011. Break our steganographic system. In IH
(LNCS, Vol. 6958). Springer, 59–70.