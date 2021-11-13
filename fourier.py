import cv2
import math
import numpy as np
from pathlib import Path
from sys import argv
FILENAME="flower.jpg"

def fft_fast(img: np.ndarray):
    dft = cv2.dft(img.astype(np.float), flags=cv2.DFT_COMPLEX_OUTPUT)
    shifted = fft_shift_fast(dft)
    dft_mag: np.ndarray = 20 * np.log(cv2.magnitude(shifted[:,:,0],shifted[:,:,1])+1)
    return dft_mag.astype(np.dtype('uint8'))

def fft_shift_fast(img: np.ndarray):
    return np.fft.fftshift(img)


if __name__ == "__main__":
    path = Path(argv[1] if len(argv) > 1 else FILENAME)
    img = cv2.imread(path.as_posix(), flags=cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("{}_orig.bmp".format(path.stem), img)
    freq: np.ndarray = fft_fast(img)
    cv2.imshow("original in grayscale", img)
    cv2.imshow("freq", freq)
    cv2.waitKey(0) # required so the images have time to be shown
    cv2.destroyAllWindows()