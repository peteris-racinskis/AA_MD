import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sys import argv
FILENAME="flower.jpg"

# trade space for time - use C arrays in memory (fast)
# instead of loops in python (very slow)
def histogram_smoothing(img: np.ndarray):
    # vector of discrete levels length n
    levels = np.arange(0, 255, 1)
    # flatten image into vector length m
    flat = img.flatten()
    # create n x m matrices holding copies of pixel and level values
    level_grid, pixel_grid = np.meshgrid(levels, flat, indexing='ij')
    # element-wise comparison
    pixel_grid = pixel_grid < level_grid
    # row-wise count, returns vector of n values
    cdf = np.count_nonzero(pixel_grid, axis=1)
    # apply the histogram normalization formula to vector
    normalized_cdf = np.rint((cdf - 1) * (1 / (img.size - 1)) * cdf.size)
    # image where each pixel maps to corresponding normalized cdf value by index
    smoothed: np.ndarray = np.take(normalized_cdf, img)
    # cast to uint8
    return smoothed.astype(np.dtype('uint8'))


def draw_histogram(img, img2):
    fig, (ax1, ax2) = plt.subplots(2)
    bins = np.linspace(0,255,100)
    ax1.hist(img.flatten(), bins=bins)
    ax2.hist(img2.flatten(), bins=bins)
    plt.show()

if __name__ == "__main__":
    path = Path(argv[1] if len(argv) > 1 else FILENAME)
    img = cv2.imread(path.as_posix(), flags=cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("{}_orig.bmp".format(path.stem), img)
    smoothed = histogram_smoothing(img)
    cv2.imshow("original in grayscale", img)
    cv2.imshow("smoothed", smoothed)
    cv2.waitKey(50) # required so the images have time to be shown
    draw_histogram(img, smoothed)
    cv2.destroyAllWindows()
    cv2.imwrite("{}_smoothed.bmp".format(path.stem), smoothed)