# Helper functions for IAU course of morphology.
# by ZHUO Xu (212138) @ SEU
# https://github.com/z0gSh1u/seu-iau

import numpy as np
import matplotlib.pyplot as plt


def readBinaryImage(path, delim=','):
    trimNotEmpty = lambda x: len(x.strip()) > 0
    with open(path, 'r') as fp:
        content = fp.read()

    rows = list(filter(trimNotEmpty, content.replace('\r\n', '\n').split('\n')))
    h = len(rows)
    row = list(filter(trimNotEmpty, rows[0].split(delim)))
    w = len(row)
    content = list(map(int, list(filter(trimNotEmpty, content.replace('\n', ',').split(delim)))))

    return np.array(content, dtype=bool).reshape(h, w)


def display(img, title=''):
    if img.dtype == bool:
        disp = np.array(img, dtype=np.uint8)
        lut = [0, 255]
        disp[disp == 0] = lut[0]
        disp[disp == 1] = lut[1]
        plt.title(title)
        plt.imshow(disp, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(img, cmap='gray')
        plt.title(title)
