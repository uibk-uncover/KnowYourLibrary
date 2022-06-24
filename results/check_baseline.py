
import numpy as np
from PIL import Image
import sys

f1,f2 = sys.argv[1:]

x1 = np.array(Image.open(f1))
x2 = np.array(Image.open(f2))

same = (x1 == x2).all()
if same:
    print("Images are the same.")
else:
    print("Images are different.")
