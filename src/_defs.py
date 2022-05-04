
from typing import Tuple

class TestContext:
    # versions to test
    versions: list = ['6b','turbo210','7','8','8a','8b','8c','8d','9','9a','9b','9c','9d','9e']
    # arbitrary version
    v_arbitrary: str = '9e'
    # chroma subsampling
    samp_factor:Tuple[ Tuple[ int,int ], Tuple[ int,int ], Tuple[ int,int ] ] = None
    use_chroma_sampling: bool = None
    # DCT method
    dct_method: str = None
    dct_method_arbitrary: str = None
    # quality
    quality: int = None
    # colorspace
    colorspace: str = None
    