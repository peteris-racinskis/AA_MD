import cv2
import numpy as np
from pathlib import Path
from sys import argv
from typing import List
FILENAME="thin-input.png"
DRAW_DELAY=0
LO=0
HI=255
DC=127
MAX_ITERS=100

# Generate the thinning hit-and-miss kernel set
# Taken from: https://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm 
# Hit checking is done by setting the required match count. 
# All don't care values set to ones that cannot appear in
# the binary image.
def kernels() -> List[np.ndarray]:
    outcrop = np.asarray([[LO,LO,LO], [DC,HI,DC], [HI,HI,HI]]).astype(np.uint8)
    corner = np.asarray([[DC,LO,LO], [HI,HI,LO], [DC,HI,DC]]).astype(np.uint8)
    kernel_set = []
    for k in outcrop, corner:
        matches = [np.count_nonzero(k != DC)] * 4
        rotations = [k, k.T, np.flip(k, axis=0), np.flip(k, axis=1)]
        kernel_set += [(rot, match) for rot, match in zip(rotations, matches)]
    return kernel_set

# image has to be binary with HI and LO values as specified above
def thin_iter(img: np.ndarray, k_set: np.ndarray) -> np.ndarray:
    n, m = img.shape
    x_range = range(1,n-1)
    y_range = range(1,m-1)
    out = np.copy(img)
    for i in x_range:
        for j in y_range:
            if out[i][j] == 0:
                continue
            for k, matches in k_set:
                window = out[i-1:i+2,j-1:j+2]
                hits = np.count_nonzero(window == k)
                if hits == matches:
                    out[i][j] = 0
    return out

def thin(img:np.ndarray, max_iters) -> np.ndarray:
    k_set = kernels()
    converged = False
    iters = 0
    previous = img
    while not converged and iters < max_iters:
        iters += 1
        result = thin_iter(previous,k_set)
        if np.count_nonzero(result - previous) == 0:
            converged = True
            print(f"Converged after {iters} iterations")
        previous = result
    return previous

def force_binary(img: np.ndarray) -> np.ndarray:
    return np.where(img == 0, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    path = Path(argv[1] if len(argv) > 1 else FILENAME)
    img = force_binary(cv2.imread(path.as_posix(), flags=cv2.IMREAD_GRAYSCALE))
    thinned = thin(img, MAX_ITERS)

    cv2.imwrite("thinning/{}_original-forced-binary.bmp".format(path.stem), img)
    cv2.imwrite("thinning/{}_thinned.bmp".format(path.stem), thinned)

    cv2.imshow("original binary image", img)
    cv2.imshow("thinned binary image", thinned)
    cv2.waitKey(DRAW_DELAY)
    cv2.destroyAllWindows()
