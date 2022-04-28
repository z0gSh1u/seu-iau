# CUDA acclerated fuzzy image processing operations
# by ZHUO Xu (212138) @ SEU
# https://github.com/z0gSh1u/seu-iau

import os
import numpy as np
import pycuda.autoinit
from pycuda.driver import In as _In, Out as _Out
from pycuda.compiler import SourceModule

GlobalNVCCOptions = [
    '-ccbin',
    # Modify this path to your MSVC compiler path.
    'D:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.29.30037/bin/Hostx86/x64/cl.exe'
]

# Compile kernel functions.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuFuzzy.cu')) as fp:
    _cuFuzzy = SourceModule(fp.read(), options=GlobalNVCCOptions)


def fuzzyEdgeExtract3x3(img, sigma):
    img = img.astype(np.int32)
    h, w = img.shape
    params = np.array([h, w, sigma]).astype(np.int32)
    res = np.zeros_like(img, dtype=np.int32)

    _cuFuzzy.get_function('fuzzyEdgeExtract3x3')(_Out(res),
                                                 _In(img),
                                                 _In(params),
                                                 block=(16, 16, 1),
                                                 grid=((h + 15) // 16, (w + 15) // 16))

    return res.astype(np.uint8)