import cv2
import numpy as np
from math import sqrt
from pathlib import Path
from typing import Tuple
from sys import argv
from fourier import pixels
FILENAME="flower.jpg"
DRAW_DELAY=0

# Forward Haar transform level and wavelet component
# [  V  ]
# [  W  ]
# where V - scaling coefficients (+ 1/sqrt(2))
#       W - difference coefficients (+- 1/sqrt(2))
# The sqrt(2)s are there to keep the magnitude of the transform 
# vector the same as that of the input vector. If this wasn't done,
# the output would balloon in size.
# The inverse Haar transform recombination matrix is the transpose of 
# the forward transform corner matrix:
# 1/sqrt(2)*[ V_inv ][ W_inv ] = 
#=[1 0 0 ... ][ 1  0 0 ... ]
# [1 0 0 ... ][-1  0 0 ... ] * 1/sqrt(2)
# [0 1 0 ... ][ 0  1 0 ... ]
# [0 1 0 ... ][ 0 -1 0 ... ]
# where V_inv - inverse coefficients of trend vector
#       W_inv - inverse coefficients of difference vector
# Once again, sqrt(2) needed to conserve vector magnitude.
def corner(N, inverse=False) -> np.ndarray:
    n, m = np.meshgrid(np.arange(0,N,1), np.arange(0,N / 2,1))
    first = n == 2 * m
    second = n == (2 * m) + 1
    V = np.where(np.logical_or(first, second), 1, 0)
    W = np.where(second, -V, V)
    corner = np.concatenate([V,W]) * 1/sqrt(2)
    return corner.T if inverse else corner

# Stage composed matrix
# [        |                ]
# [ corner |       0        ]
# [ ------ | -------------- ]
# [        |                ]
# [        |                ]
# [   0    |    Identity    ]
# [        |                ]
# [        |                ]
# [        |                ]
# where corner - inverse or direct corner matrix
# the size of the corner depends on the order of this
# transform matrix - for direct, it starts with heigh
# N / 2 and width M, for inverse the sizes are transposed.
# Then with every successive application of the transform
# the height and width  of the corner are cut in half, the 
# remainder of the matrix forming I.
def compose_stage(n, order, inverse=False) -> np.ndarray:
    scaler = 2 ** order
    N = int(n / scaler)
    base = corner(N, inverse)
    diff = n-N
    padded = np.pad(base, ((0, diff), (0, diff)))
    id = np.pad(np.identity(diff), ((N, 0), (N, 0)))
    composed = padded + id
    return composed

# For an order n transform, both axes of the image must divide 2^(n+1)
# Since all stage matrices are linear transformations, they can be multiplied
# together to get the transform along one axis.
def discrete_wavelet_transform(img: np.ndarray, order, inverse=False) -> Tuple[np.ndarray]:
    n, m = img.shape
    Fx = np.identity(n)
    Fy = np.identity(m)
    # reverse order on inverse pass
    iterator = range(0, order+1) if not inverse else range(order,-1,-1)
    for i in iterator:
        Fx = compose_stage(n, i, inverse) @ Fx
        Fy = compose_stage(m, i, inverse) @ Fy
    result = Fx @ img @ Fy.T
    return result.astype(np.uint8) if inverse else result

def threshold_denoise(img: np.ndarray, threshold) -> np.ndarray:
    mask = np.abs(img) > threshold
    return np.where(mask, img, 0)


if __name__ == "__main__":
    path = Path(argv[1] if len(argv) > 1 else FILENAME)
    img = cv2.imread(path.as_posix(), flags=cv2.IMREAD_GRAYSCALE)
    haar = discrete_wavelet_transform(img, 3)
    clipped = threshold_denoise(haar, 20)
    pure_inv = discrete_wavelet_transform(haar, 3, inverse=True)
    clip_inv = discrete_wavelet_transform(clipped, 3, inverse=True)
    cv2.imshow("original in grayscale", img)
    cv2.imshow("haar wavelet transform pixels", pixels(haar))
    cv2.imshow("haar wavelet transform pixels clipped", pixels(clipped))
    cv2.imshow("pure inverse", pure_inv)
    cv2.imshow("clip inverse", clip_inv)
    cv2.waitKey(DRAW_DELAY)