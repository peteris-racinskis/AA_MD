import cv2
import numpy as np
from pathlib import Path
from sys import argv
from gauss import gauss_blur_fast
from typing import List
FILENAME="flower.jpg"
DRAW_DELAY=0
THRESHOLD=50
HI=255
LO=0

def sobel_kernel():
    Ky = np.asarray([[1,2,1],[0,0,0],[-1,-2,-1]])
    Kx = Ky.T
    return Kx, Ky

def laplace_kernel(diagonal=True):
    if diagonal:
        K = np.asarray([[1,1,1],[1,-8,1],[1,1,1]])
    else:
        K = np.asarray([[0,1,0],[1,-4,1],[0,1,0]])
    return K

# Since we're doing edge detection in a large image, the outer
# row/column of the image can simply be ignored, no normalization
# required. All kernels are 3x3 which makes this function simpler.
def convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    output = np.zeros(img.shape)
    n, m = img.shape
    x_range = range(1,n-1)
    y_range = range(1,m-1)
    for i in x_range:
        for j in y_range:
            window = img[i-1:i+2,j-1:j+2]
            output[i][j] = np.sum(window * kernel)
    return output

def sobel(img: np.ndarray) -> List[np.ndarray]:
    Kx, Ky = sobel_kernel()
    Gx = convolve(img, Kx)
    Gy = convolve(img, Ky)
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    return [Gx, Gy, G]

def laplace(img: np.ndarray, diagonal=True) -> np.ndarray:
    K = laplace_kernel(diagonal)
    return cut_negative(convolve(img, K))

# center and cast to 8-bit for display
def normalize(images: List[np.ndarray]) -> np.ndarray:
    output = []
    for img in images:
        output.append((img + 127).astype(np.uint8))
    return output

# This isn't described usually but seems to be implicitly
# done by cv2's and other implementations of the filters.
def cut_negative(img: np.ndarray) -> np.ndarray:
    return np.where(img > 0, img, 0)

# Existing opencv implementation of laplace to make sure
# the one I wrote is behaving similarly. Most real 
# implementations of the Laplace operator use a combined
# Laplacian-Gaussian kernel so a separate blur step is not
# required, and the result is slightly different.
def opencv_laplace(img: np.ndarray) -> np.ndarray:
    return cv2.Laplacian(img, cv2.CV_8U)

# Simple binary thresholding
def binary_edge(img: np.ndarray, thresh=THRESHOLD) -> np.ndarray:
    return np.where(img > thresh, HI, LO).astype(np.uint8)


if __name__ == "__main__":
    path = Path(argv[1] if len(argv) > 1 else FILENAME)
    img = cv2.imread(path.as_posix(), flags=cv2.IMREAD_GRAYSCALE)
    blurred_lo = gauss_blur_fast(img, radius=1.8)
    # for sobel edge detect
    blurred_hi = gauss_blur_fast(img, radius=3)
    
    # Kernel effect comparison
    laplace_blur_zero = laplace(blurred_lo)
    laplace_pure, = normalize([laplace(img)])
    laplace_blur, = normalize([laplace_blur_zero])
    laplace_cv2 = opencv_laplace(img)
    laplace_cv2_norm, = normalize([laplace_cv2])
    sobel_x, sobel_y, sobel_cmb = normalize(sobel(img))
    sobel_x_b, sobel_y_b, sobel_cmb_b = normalize(sobel(blurred_lo))
    
    # Edge detector comparison
    laplace_cv2_edge = binary_edge(laplace_cv2)
    laplace_blur_edge = binary_edge(laplace_blur_zero)
    sobel_edge = binary_edge(sobel(blurred_hi)[2])

    cv2.imwrite("edge-kernel/{}_original.bmp".format(path.stem), img)
    cv2.imwrite("edge-kernel/{}_laplace-orig.bmp".format(path.stem), laplace_pure)
    cv2.imwrite("edge-kernel/{}_laplace-blur.bmp".format(path.stem), laplace_blur)
    cv2.imwrite("edge-kernel/{}_sobel-v-orig.bmp".format(path.stem), sobel_x)
    cv2.imwrite("edge-kernel/{}_sobel-h-orig.bmp".format(path.stem), sobel_y)
    cv2.imwrite("edge-kernel/{}_sobel-c-orig.bmp".format(path.stem), sobel_cmb)
    cv2.imwrite("edge-kernel/{}_sobel-v-blur.bmp".format(path.stem), sobel_x_b)
    cv2.imwrite("edge-kernel/{}_sobel-h-blur.bmp".format(path.stem), sobel_y_b)
    cv2.imwrite("edge-kernel/{}_sobel-c-blur.bmp".format(path.stem), sobel_cmb_b)
    cv2.imwrite("edge-kernel/{}_edges-laplace-cv2.bmp".format(path.stem), laplace_cv2_edge)
    cv2.imwrite("edge-kernel/{}_edges-laplace-blur.bmp".format(path.stem), laplace_blur_edge)
    cv2.imwrite("edge-kernel/{}_edges-sobel.bmp".format(path.stem), sobel_edge)

    cv2.imshow("original in grayscale", img)
    cv2.imshow("original in grayscale", img)
    cv2.imshow("laplace kernel - orig", laplace_pure)
    cv2.imshow("laplace kernel - blur", laplace_blur)
    cv2.imshow("sobel combined - orig", sobel_cmb)
    cv2.imshow("sobel vertical - blur", sobel_x_b)
    cv2.imshow("sobel horizontal - blur", sobel_y_b)
    cv2.imshow("sobel combined - blur", sobel_cmb_b)
    cv2.imshow("laplace cv2", laplace_cv2_norm)
    cv2.imshow("laplace cv2 edge", laplace_cv2_edge)
    cv2.imshow("laplace blur edge", laplace_blur_edge)
    cv2.imshow("sobel combined edge", sobel_edge)
    cv2.waitKey(DRAW_DELAY)
    cv2.destroyAllWindows()
