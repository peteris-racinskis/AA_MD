import cv2
import numpy as np
from math import sqrt
from pathlib import Path
from typing import Tuple
from sys import argv
from fourier import pixels
FILENAME="flower-noise.jpg"
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

# For an order n transform, both axes of the image must be divisible by 2^(n+1)
# Since all stage matrices are linear transformations, they can be multiplied
# together to get the transform along one axis.
# This implementation is based on the 1-dimensional Haar wavelet transform discussed
# here: http://dsp-book.narod.ru/PWSA/8276_01.pdf 
# It already gives the matrix equations but for some reason tries to pretend
# no linear algebra is involved. In any case, I took the liberty of deriving
# the general case nth step Haar transform matrix as well as the complete
# form that involves combining these matrices myself. Extending the 1-dimensional
# case to 2 dimensions was trivial, as was constructing the inverse algorithm.
def discrete_wavelet_transform(img: np.ndarray, order, inverse=False) -> Tuple[np.ndarray]:
    n, m = img.shape
    Fx = np.identity(n)
    Fy = np.identity(m)
    # reverse order on inverse pass because Matrix multiplication is not
    # commutative.
    iterator = range(0, order+1) if not inverse else range(order,-1,-1)
    for i in iterator:
        Fx = compose_stage(n, i, inverse) @ Fx
        Fy = compose_stage(m, i, inverse) @ Fy
    result = Fx @ img @ Fy.T
    return result.astype(np.uint8) if inverse else result


# Hard threshold. Discontinuities cause ringing.
def hard_threshold(img: np.ndarray, threshold) -> np.ndarray:
    mask = np.abs(img) > threshold
    return np.where(mask, img, 0)

# Soft threshold - to remove discontinuities. Not perfect, but cuts
# down on the ringing immensely. A smarter thresholding method should
# be used to get rid of it entirely but this is good enough for a proof
# of concept.
# Equation from 
# https://scholarworks.sjsu.edu/cgi/viewcontent.cgi?article=4892&context=etd_theses
def soft_threshold(img: np.ndarray, threshold) -> np.ndarray:
    mask = np.abs(img) > threshold
    img = np.sign(img) * (np.abs(img)-threshold)
    return np.where(mask, img, 0)


if __name__ == "__main__":
    path = Path(argv[1] if len(argv) > 1 else FILENAME)
    img = cv2.imread(path.as_posix(), flags=cv2.IMREAD_GRAYSCALE)

    haar = discrete_wavelet_transform(img, 3)
    clipped = hard_threshold(haar, 10)
    smoothed = soft_threshold(haar, 10)
    pure_inv = discrete_wavelet_transform(haar, 3, inverse=True)
    clip_inv = discrete_wavelet_transform(clipped, 3, inverse=True)
    soft_inv = discrete_wavelet_transform(smoothed, 3, inverse=True)

    cv2.imwrite("wavelet/{}_original.bmp".format(path.stem), img)
    cv2.imwrite("wavelet/{}_wavelet_transform.bmp".format(path.stem), pixels(haar))
    cv2.imwrite("wavelet/{}_restored.bmp".format(path.stem), pure_inv)
    cv2.imwrite("wavelet/{}_restored-hard-thresh.bmp".format(path.stem), clip_inv)
    cv2.imwrite("wavelet/{}_restored-soft-thresh.bmp".format(path.stem), soft_inv)

    cv2.imshow("original in grayscale", img)
    cv2.imshow("haar wavelet transform", pixels(haar))
    cv2.imshow("pure inverse", pure_inv)
    cv2.imshow("clip inverse", clip_inv)
    cv2.imshow("smooth inverse", soft_inv)
    cv2.waitKey(DRAW_DELAY)
    cv2.destroyAllWindows()