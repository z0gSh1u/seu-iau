import numpy as np
import matplotlib.pyplot as plt

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
