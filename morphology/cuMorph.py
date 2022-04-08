# CUDA acclerated morphology operations
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
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuMorph.cu')) as fp:
    _cuMorph = SourceModule(fp.read(), options=GlobalNVCCOptions)

### Binary ###


def union(a, b):
    assert np.array_equal(a.shape, b.shape)
    return np.logical_or(a, b)


def intersect(a, b):
    assert np.array_equal(a.shape, b.shape)
    return np.logical_and(a, b)


def same(a, b):
    return np.array_equal(a, b)


def binaryComplement(img):
    return np.logical_not(img)


def binaryErode(img, se):
    img = img.astype(np.int32)
    se = se.astype(np.int32)
    h, w = img.shape
    hw = np.array([h, w, *se.shape], dtype=np.int32)
    res = np.zeros_like(img, dtype=np.int32)

    _cuMorph.get_function('binaryErode')(_Out(res),
                                         _In(img),
                                         _In(se),
                                         _In(hw),
                                         block=(16, 16, 1),
                                         grid=((h + 15) // 16, (w + 15) // 16))

    return res.astype(bool)


def binaryDilate(img, se):
    return binaryComplement(binaryErode(binaryComplement(img), np.flipud(np.fliplr(se))))


def binaryOpen(img, se):
    return binaryDilate(binaryErode(img, se), se)


def binaryClose(img, se):
    return binaryErode(binaryDilate(img, se), se)


def hitOrMiss(img, se1, se2):
    t1 = binaryErode(img, se1)
    t2 = binaryErode(binaryComplement(img), se2)

    return intersect(t1, t2)


def convexHull(img, se1, se2):
    Ich = np.array(img)
    Ichs = [np.array(Ich)]
    i = 0

    while True:
        Ich = union(hitOrMiss(Ich, se1, se2), Ich)
        if same(Ich, Ichs[-1]):
            break
        Ichs.append(np.array(Ich))
        i += 1

    return Ich, i


### Binary ###

### --------------------------------------- ###

### Grayscale ###


def grayscaleComplement(img):
    return np.array(-img, dtype=np.int32)


def grayscaleErode(img, se):
    img = img.astype(np.int32)
    se = se.astype(np.int32)
    h, w = img.shape
    hw = np.array([h, w, *se.shape], dtype=np.int32)
    res = np.zeros_like(img, dtype=np.int32)

    _cuMorph.get_function('grayscaleErode')(_Out(res),
                                            _In(img),
                                            _In(se),
                                            _In(hw),
                                            block=(16, 16, 1),
                                            grid=((h + 15) // 16, (w + 15) // 16))

    return res.astype(np.int32)


def grayscaleDilate(img, se):
    return grayscaleComplement(grayscaleErode(grayscaleComplement(img), np.flipud(np.fliplr(se))))


def grayscaleOpen(img, se):
    return grayscaleDilate(grayscaleErode(img, se), se)


def grayscaleClose(img, se):
    return grayscaleErode(grayscaleDilate(img, se), se)


def tophat(img, se):
    return img.astype(np.int32) - grayscaleOpen(img, se)

# `same` can be borrowed from binary.

### Grayscale ###