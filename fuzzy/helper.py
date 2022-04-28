import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def readImageUint8(filepath, delim=','):
    with open(filepath, 'r') as fp:
        content = fp.read()

    content = content.strip().replace('\r\n', '\n').split('\n')
    img = []
    for line in content:
        row = list(map(int, line.strip().split(delim)))
        img.append(row)
    return np.array(img, dtype=np.uint8)


def display(img, title=''):
    plt.imshow(img, cmap='gray')
    plt.title(title)


def fuzzyEdgeExtract3x3Py(img, muWhite, muBlack, muZeroD):
    h, w = img.shape
    ksize = 3
    pad = ksize // 2
    z5 = np.arange(0, 255)
    res = np.zeros_like(img)
    for i in trange(pad, h - pad):
        for j in range(pad, w - pad):
            patch = img[i - pad:i + pad + 1, j - pad:j + pad + 1]
            diff = patch.flatten().astype(np.int16)
            diff = diff - diff[4]  # -z5
            muZeroDs = [-1, *[muZeroD(x) for x in diff]]
            mu1Z5D2D6 = [min(muWhite(x), muZeroDs[2], muZeroDs[6]) for x in z5]
            mu2Z5D6D8 = [min(muWhite(x), muZeroDs[6], muZeroDs[8]) for x in z5]
            mu3Z5D8D4 = [min(muWhite(x), muZeroDs[8], muZeroDs[4]) for x in z5]
            mu4Z5D4D2 = [min(muWhite(x), muZeroDs[4], muZeroDs[2]) for x in z5]
            mu5Z5 = [muBlack(x) for x in z5]
            muZ5 = [max(mu1Z5D2D6[x], mu2Z5D6D8[x], mu3Z5D8D4[x], mu4Z5D4D2[x], mu5Z5[x]) for x in range(len(z5))]
            muZ5 = np.array(muZ5)
            z5Apply = np.sum(z5 * muZ5) / np.sum(muZ5)
            z5Apply = np.round(z5Apply).clip(0, 255).astype(np.uint8)
            res[i, j] = z5Apply

    return res