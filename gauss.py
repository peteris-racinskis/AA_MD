import cv2
import math
import numpy as np
from pathlib import Path
from sys import argv
import time
FILENAME="flower.jpg"

# radius - pixels from origin
# sigma - standard deviation of the Gaussian distribution
def generate_kernel(radius: int, sigma: float) -> np.ndarray:
    m = 2 * radius if radius % 2 == 1 else 2 * radius + 1
    interval = np.arange(1,m,1, dtype="float")
    x,y = np.meshgrid(interval,interval)
    distance = (x - radius) ** 2 + (y - radius) ** 2
    kernel = np.exp((distance * -1) / (2 * (sigma ** 2))) / \
         (2 * math.pi * (sigma ** 2))
    return kernel

# for kernels reaching over edges, normalize
# to sum to the same value as the regular kernel.
# this particular method works only with radially
# symmetric kernels. The order of the indices doesn't
# actually matter, only their sum - the result is scalar
# anyway. P could be set equal to 1, it should be close
def kernel_normalize(kernel, x, y):
    P = np.sum(kernel)
    m = len(kernel) + 1
    xint = np.flip(np.arange(1-x,m-x,1, dtype="int"))
    yint = np.flip(np.arange(1-y,m-y,1, dtype="int"))
    a, b = np.meshgrid(xint, yint)
    mask = np.logical_and(a > 0,b > 0)
    return P / np.sum(kernel * mask.astype(float))

# Since this is a homework assignment, need to show
# that I understand the convolution process, even
# if the iteration code is very slow. Precompute the kernel
# normalizations because creating/destroying python stack
# frames is extreeeeeemly slow.
def gauss_blur_slow(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = len(kernel) // 2
    n,m = img.shape
    output = np.zeros(img.shape)
    img = np.pad(img, ((radius,radius),(radius,radius)))
    norm = {}
    for x in range(radius+1):
        for y in range(radius+1):
            norm[x,y] = kernel_normalize(kernel, x, y)
    for i in range(n):
        for j in range(m):
            k = norm[-min(i-radius,n-i-radius,0), -min(j-radius,m-j-radius,0)]
            window = img[i:i+(2*radius)+1, j:j+(2*radius)+1]
            output[i,j] = k * np.sum(window*kernel)
    return output.astype(np.dtype('uint8'))

# for comparison, how much faster is the opencv implementation?
# roughly 1000x
def gauss_blur_fast(img, kernel=None) -> np.ndarray:
    if kernel is None:
        kernel = generate_kernel(3, 1)
    return cv2.filter2D(img,-1,kernel)

if __name__ == "__main__":
    path = Path(argv[1] if len(argv) > 1 else FILENAME)
    radius = int(argv[2]) if len(argv) > 2 and argv[2].isdigit() else 3
    fast = "fast" in argv
    img = cv2.imread(path.as_posix(), flags=cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("gauss/{}_original.bmp".format(path.stem), img)
    kernel = generate_kernel(radius, radius / 3)
    t0 = time.time()
    if not fast:
        blurred = gauss_blur_slow(img, kernel)
        cv2.imwrite("gauss/{}_blurred-slow.bmp".format(path.stem), blurred)
    else:
        blurred = gauss_blur_fast(img, kernel)
        cv2.imwrite("gauss/{}_blurred-fast.bmp".format(path.stem), blurred)
    t1 = time.time()
    print(t1-t0)
    cv2.imshow("original in grayscale", img)
    cv2.imshow("blurred", blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()