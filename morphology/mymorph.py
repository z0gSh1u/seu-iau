# CUDA acclerated morphology operations

import numpy as np
import pycuda.autoinit
from pycuda.driver import In as _In, Out as _Out
from pycuda.compiler import SourceModule

GlobalNVCCOptions = [
    '-ccbin',
    'D:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.29.30037/bin/Hostx86/x64/cl.exe'
]

CUDA_binaryErode = SourceModule('''
    __global__ void binaryErode(int* res, const int* img, const int *se, const int* hw) {
        int i = (blockIdx.x * blockDim.x) + threadIdx.x, j = (blockIdx.y * blockDim.y) + threadIdx.y;
        if (i >= srcH || j >= srcW) {
            return;
        }
        int hImg = hw[0], wImg = hw[1], hSE = hw[2], wSE = hw[3];
        int rhSE = hSE / 2, rwSE = wSE / 2; // radius
        int r, c, pix, valid = 1;

        for (int m = -rhSE; m <= rhSE; m++) {
            for (int n = -rwSE; n <= rwSE; n++) {
                r = i + m; c = j + n;
                if (r < 0 || r >= hImg || c < 0 || c >= wImg) {
                    pix = 1;
                } else {
                    pix = img[r * wImg + c];
                }
                if (!pix && se[(m + rwSE) * wSE + (n + rcSE)]) {
                    valid = 0;
                    break;
                }
            }
        }
        res[i * wImg + j] = valid && img[i * wImg + j];
    }
''')


def binaryErode(img, se):
    f = CUDA_binaryErode.get_function('binaryErode')
    res = np.zeros_like(img)
    hw = np.array([*img.shape, *se.shape])
    f(_Out(res), _In(img), _In(se), _In(hw)) # TODO block, grid
    return res