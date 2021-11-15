from fourier import *

def motion_blur_psf(img: np.ndarray) -> np.ndarray:
    pass

def gaussian_psf(img: np.ndarray) -> np.ndarray:
    pass

def convolve(img, psf):
    f, h = ft_fast(img), ft_fast(psf)
    return ift_fast(f * h)
