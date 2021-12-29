import cv2
import numpy as np
from pathlib import Path
from sys import argv
from gauss import gauss_blur_fast
from typing import List
FILENAME="flower.jpg"
DRAW_DELAY=0

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
    return convolve(img, K)

# center and cast to 8-bit for display
def normalize(images: List[np.ndarray]) -> np.ndarray:
    output = []
    for img in images:
        output.append((img + 127).astype(np.uint8))
    return output


if __name__ == "__main__":
    path = Path(argv[1] if len(argv) > 1 else FILENAME)
    img = cv2.imread(path.as_posix(), flags=cv2.IMREAD_GRAYSCALE)
    blurred = gauss_blur_fast(img)
    
    laplace_pure, = normalize([laplace(img)])
    laplace_blur, = normalize([laplace(blurred)])
    sobel_x, sobel_y, sobel_cmb = normalize(sobel(img))
    sobel_x_b, sobel_y_b, sobel_cmb_b = normalize(sobel(blurred))

    cv2.imwrite("predictive/{}_original.bmp".format(path.stem), img)
    cv2.imwrite("predictive/{}_laplace-orig.bmp".format(path.stem), laplace_pure)
    cv2.imwrite("predictive/{}_laplace-blur.bmp".format(path.stem), laplace_blur)
    cv2.imwrite("predictive/{}_sobel-v-orig.bmp".format(path.stem), sobel_x)
    cv2.imwrite("predictive/{}_sobel-h-orig.bmp".format(path.stem), sobel_y)
    cv2.imwrite("predictive/{}_sobel-c-orig.bmp".format(path.stem), sobel_cmb)
    cv2.imwrite("predictive/{}_sobel-v-blur.bmp".format(path.stem), sobel_x_b)
    cv2.imwrite("predictive/{}_sobel-h-blur.bmp".format(path.stem), sobel_y_b)
    cv2.imwrite("predictive/{}_sobel-c-blur.bmp".format(path.stem), sobel_cmb_b)

    cv2.imshow("original in grayscale", img)
    cv2.imshow("laplace edges - orig", laplace_pure)
    cv2.imshow("laplace edges - blur", laplace_blur)
    cv2.imshow("sobel vertical - orig", sobel_x)
    cv2.imshow("sobel horizontal - orig", sobel_y)
    cv2.imshow("sobel combined - orig", sobel_cmb)
    cv2.imshow("sobel vertical - blur", sobel_x_b)
    cv2.imshow("sobel horizontal - blur", sobel_y_b)
    cv2.imshow("sobel combined - blur", sobel_cmb_b)
    cv2.waitKey(DRAW_DELAY)
