
from typing import Tuple

class TestContext:
    # versions to test
    versions: list = ['6b','turbo210','7','8','8a','8b','8c','8d','9','9a','9b','9c','9d','9e']
    # arbitrary version
    v_arbitrary: str = '9e'
    # chroma subsampling
    samp_factor:Tuple[ 3*Tuple[ 2*int ] ] = None
    use_chroma_sampling: bool = None
    
