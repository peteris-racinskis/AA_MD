import cv2
import math
import numpy as np
from pathlib import Path
from sys import argv
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
# Fkl - final result (or restored image)
# Fkl = Bkn*Fmn*Cml
# The equations were taken from:
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/fourier.htm
# The matrix form I derived myself though I am certain
# there are materials out there describing it - it's the
# obvious way to express this algorithm if fast matrix
# multiplication tools are available.
def dft_matrix(img: np.ndarray, inverse=False) -> np.ndarray:
    sign = 1 if inverse else -1
    Fmn = img.astype(dtype='complex')
    N, M = Fmn.shape
    k, n = np.meshgrid(np.arange(0,N,1), np.arange(0,N,1))
    l, m = np.meshgrid(np.arange(0,M,1), np.arange(0,M,1))
    Bkn = np.cos((sign / N) * (2 * math.pi * n * k)) + 1j * np.sin((sign / N) * (2 * math.pi * n * k))
    Cml = np.cos((sign / M) * (2 * math.pi * m * l)) + 1j * np.sin((sign / M) * (2 * math.pi * m * l))
    Fkl = Bkn @ Fmn @ Cml
    return np.abs(Fkl) / (N * M) if inverse else Fkl

def inverse_shifted(mat: np.ndarray) -> np.ndarray:
    return dft_matrix(ft_shift_slow(mat, inverse=True), inverse=True)

def pixels(mat: np.ndarray, inverse=False) -> np.ndarray:
    return (20 * np.log(np.abs(mat)+1) if not inverse else mat).astype(np.dtype('uint8'))

# convolution in the space domain is elementwise multiplication
# in the spatial frequency domain.
def convolve_transforms(img: np.ndarray, mask: np.ndarray):
    return img * mask

def ft_fast(img: np.ndarray) -> np.ndarray:
    # Could also use CV2's implementation - returns complex numbers as
    # arrays rather than tuples
    #dft = cv2.dft(img.astype(np.float), flags=cv2.DFT_COMPLEX_OUTPUT)
    #dft_complex = dft[...,0] + 1j * dft[...,1]
    dft_complex = np.fft.fft2(img)
    shifted = ft_shift_slow(dft_complex)
    return shifted

def ift_fast(img: np.ndarray, shifted=True) -> np.ndarray:
    ft = ft_shift_slow(img, inverse=True) if shifted else img
    ift = np.fft.ifft2(ft)
    return np.abs(ift)

def ft_slow(img: np.ndarray) -> np.ndarray:
    dft = dft_matrix(img)
    shifted = ft_shift_slow(dft)
    return shifted

# np.fft has a fftshift function to relocate the corners to the center
# I wrote my own to show that I understand what it's doing.
def ft_shift_slow(img: np.ndarray, inverse=False) -> np.ndarray:
    s = -1 if inverse else 1
    M, N = img.shape
    return np.roll(img, (s * M // 2, s * N // 2), (0,1))

# Make everything outside the radius = 0
def lowpass_filter(shape, radius):
    N, M = shape
    mrange = np.concatenate((np.arange((-M // 2) + 1, 1, 1), np.arange(M % 2, M // 2, 1)))
    nrange = np.concatenate((np.arange((-N // 2) + 1, 1, 1), np.arange(N % 2, N // 2, 1)))
    mdist, ndist = np.meshgrid(mrange, nrange)
    cutoff = mdist ** 2 + ndist ** 2 <= radius ** 2
    return cutoff

if __name__ == "__main__":

    path = Path(argv[1] if len(argv) > 1 else FILENAME)
    radius = int(argv[2]) if len(argv) > 2 and argv[2].isdigit() else 20
    fast = "fast" in argv

    speed = "fast" if fast else "slow"
    img = cv2.imread(path.as_posix(), flags=cv2.IMREAD_GRAYSCALE)
    N, M = img.shape
    cv2.imwrite("fourier/{}_original.bmp".format(path.stem), img)
    mask = lowpass_filter(img.shape, radius)

    t0 = time.time()
    if not fast:
        raw = ft_slow(img)
    else:
        raw = ft_fast(img)
    t1 = time.time()
    print("time: {}".format(t1-t0))
    inv = inverse_shifted(raw)
    filtered = convolve_transforms(raw, mask)
    inv_filtered = inverse_shifted(filtered)
    test = ift_fast(raw)

    cv2.imwrite("fourier/{}_transform-{}.bmp".format(path.stem, speed), pixels(raw))
    cv2.imwrite("fourier/{}_transform-filtered-{}.bmp".format(path.stem, speed), pixels(filtered))
    cv2.imwrite("fourier/{}_restored-{}.bmp".format(path.stem, speed), pixels(inv, True))
    cv2.imwrite("fourier/{}_restored-filtered-{}.bmp".format(path.stem, speed), pixels(inv_filtered, True))
    
    cv2.imshow("original in grayscale", img)
    cv2.imshow("freq", pixels(raw))
    cv2.imshow("inv", pixels(inv, True))
    cv2.imshow("filtered", pixels(filtered))
    cv2.imshow("inv filtered", pixels(inv_filtered, True))
    cv2.imshow("test", pixels(test, True))
    cv2.waitKey(0)
    cv2.destroyAllWindows()