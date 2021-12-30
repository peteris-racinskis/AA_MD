import cv2
import numpy as np
from pathlib import Path
from sys import argv
from math import cos, sin, radians, sqrt
from thinning import force_binary
from fourier import pixels
FILENAME="hough-input-interrupted.png"
DRAW_DELAY=0
LO=0
HI=255
DC=127
eps = 0.000001

# a very sub-otpimal way of doing this but saves tons of 
# debugging time - just solve a linear system of equations 
# for each theta and point to find r rather than trying to
# come up with a general geometric rule for doing this.
def radius(x1, x2, theta):
    r1, r2 = cos(theta), sin(theta)
    coefficients = np.asarray([
        [r1, 1, 0],
        [r2, 0, 1],
        [0, r1, r2]
    ])
    ordinates = np.asarray([
        x1,
        x2,
        0
    ])
    k, _, __ = np.linalg.solve(coefficients, ordinates)
    return int(k)

# again, slow but it works.
def discretize(value, limit, bins):
    return int(min((value * bins) / limit, bins-1))

def hough_transform(img: np.ndarray, resolution = 500) -> np.ndarray:
    n,m = img.shape
    r_limit = int(sqrt(n ** 2 + m ** 2)) + 1
    t_limit = 180
    hough_space = np.zeros((resolution, resolution))
    for i in range(n):
        for j in range(m):
            if img[i][j] == HI:
                t_range = np.linspace(0, t_limit, resolution)
                for t in t_range:
                    r = radius(i,j,radians(t))
                    r_bin = int(discretize(r, r_limit, resolution / 2) + resolution / 2)
                    t_bin = discretize(t, t_limit, resolution)
                    hough_space[r_bin][t_bin] += 1
    return hough_space

def normalize_to_8bit(img: np.ndarray) -> np.ndarray:
    scaler = 255 / np.max(img)
    return (img * scaler).astype(np.uint8)

def onlymax(img: np.ndarray) -> np.ndarray:
    max_value = np.max(img)
    return np.where(img == max_value, 255, 0).astype(np.uint8)

if __name__ == "__main__":
    path = Path(argv[1] if len(argv) > 1 else FILENAME)
    img = force_binary(cv2.imread(path.as_posix(), flags=cv2.IMREAD_GRAYSCALE))

    lines = normalize_to_8bit(hough_transform(img))

    cv2.imshow("original binary image", img)
    cv2.imshow("hough transform", lines)
    cv2.waitKey(DRAW_DELAY)
    cv2.destroyAllWindows()
