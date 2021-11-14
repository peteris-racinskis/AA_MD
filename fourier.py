import cv2
import math
import numpy as np
from pathlib import Path
from sys import argv
from typing import Tuple
import time
FILENAME="flower.jpg"

# Implement straightforward DFT rather than FFT because
# FFT is quite a bit more complicated to deal with due
# to the 2^n structure of the recursive solution.
# Express summation as series of matrix operations
# Not taking advantage of the complex number representation
# in numpy would slow this code down a lot so I'm using it.
# Fmn - original image or transform
# Bnk - coefficient matrix (step 1)
# Cml - coefficient matrix (step 2)
# Pmk - intermediate result (one axis fourier)
# Pkm - transposed above
# Fkl - final result (or restored image)
# Fkl = (Fmn*Bnk)^T*Cml
# The equations were taken from:
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/fourier.htm
# The matrix form I derived myself though I am certain
# there are materials out there describing it - it's the
# obvious way to implement this algorithm if fast matrix
# multiplication tools are available.
def dft_matrix(img: np.ndarray, inverse=False) -> np.ndarray:
    sign = 1 if inverse else -1
    if inverse:
        Fmn = img[...,0] + 1j * img[...,1]
    else:
        Fmn =  img.astype(dtype='complex')
    N, M = Fmn.shape
    k, n = np.meshgrid(np.arange(0,N,1), np.arange(0,N,1))
    l, m = np.meshgrid(np.arange(0,M,1), np.arange(0,M,1))
    Bnk = np.cos((sign / N) * (2 * math.pi * n * k)) + 1j * np.sin((sign / N) * (2 * math.pi * n * k))
    Cml = np.cos((sign / M) * (2 * math.pi * m * l)) + 1j * np.sin((sign / M) * (2 * math.pi * m * l))
    Pkm = (Fmn.T @ Bnk).T
    Fkl = Pkm @ Cml
    return np.abs(Fkl) / (N * M) if inverse else Fkl

def convolve(img: np.ndarray, mask: np.ndarray):
    pass

def ft_fast(img: np.ndarray):
    dft = cv2.dft(img.astype(np.float), flags=cv2.DFT_COMPLEX_OUTPUT)
    shifted = ft_shift_fast(dft)
    dft_mag: np.ndarray = 20 * np.log(cv2.magnitude(shifted[...,0],shifted[...,1])+1)
    return dft_mag.astype(np.dtype('uint8'))

def ft_shift_fast(img: np.ndarray):
    return np.fft.fftshift(img)

def ft_slow(img: np.ndarray) -> np.ndarray:
    dft = dft_matrix(img)
    shifted = ft_shift_slow(dft)
    dft_mag: np.ndarray = 20 * np.log(np.abs(shifted)+1)
    return dft_mag.astype(np.dtype('uint8'))# inverse.astype(np.dtype('uint8'))

def ft_shift_slow(img: np.ndarray) -> np.ndarray:
    M, N = img.shape
    return np.roll(img, (M // 2, N // 2), (0,1))


if __name__ == "__main__":
    path = Path(argv[1] if len(argv) > 1 else FILENAME)
    img = cv2.imread(path.as_posix(), flags=cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("{}_orig.bmp".format(path.stem), img)
    t0 = time.time()
    freq = ft_slow(img)
    t1 = time.time()
    freq_fast: np.ndarray = ft_fast(img)
    t2 = time.time()
    print("slow: {}".format(t1-t0))
    print("fast: {}".format(t2-t1))
    cv2.imshow("original in grayscale", img)
    cv2.imshow("freq", freq)
    #cv2.imshow("inv", inv)
    cv2.imshow("freq fast", freq_fast)
    cv2.waitKey(0) # required so the images have time to be shown
    cv2.destroyAllWindows()