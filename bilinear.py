import cv2
import numpy as np
from pathlib import Path
from sys import argv
FILENAME="flower.jpg"

# construct fractional interpolation matrix
# Since this is linear interpolation it can be represented
# as a linear transformation, specifically with columns in the 
# form:
# [0, .., 0, 1/4, 2/4, 3/4, 1, 3/4, 2/4, 1/4, 0, ..] with m=1 being
# the 4nth element (with 0 indexing)
# This is trivial to generalize to any scale so I did just that.
def interp_matrix(N: int, scale: int) -> np.ndarray:
    M = scale * N
    n, m = np.meshgrid(np.arange(0,N,1), np.arange(0,M,1))
    signs = m <= scale * n
    signs = np.where(signs, 1, -1)
    fractions = (scale + signs * (m - scale * n)) / scale
    return np.where(fractions > 0, fractions, 0)

# apply interpolation matrix along each axis using a matrix
# representation. I came up with this particular solution
# myself but once again it's so straightforward that I can't
# imagine it's not already described somewhere.
def bilinear(matrix: np.ndarray, scale: int) -> np.ndarray:
    M, N = matrix.shape
    c1, c2 = interp_matrix(M, scale), interp_matrix(N, scale).T
    return (c1 @ matrix @ c2).astype(np.dtype('uint8'))


if __name__ == "__main__":
    path = Path(argv[1] if len(argv) > 1 else FILENAME)
    scale = int(argv[2]) if len(argv) > 2 and argv[2].isdigit() else 4
    img = cv2.imread(path.as_posix(), flags=cv2.IMREAD_GRAYSCALE)
    upscaled = bilinear(img, scale)
    cv2.imwrite("bilinear/{}_original.bmp".format(path.stem), img)
    cv2.imwrite("bilinear/{}_upscaled.bmp".format(path.stem), upscaled)
    cv2.imshow("original in grayscale", img)
    cv2.imshow("upscaled in grayscale", upscaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()