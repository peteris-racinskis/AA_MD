import cv2
import numpy as np
from pathlib import Path
from sys import argv
from typing import List, Tuple
from hist import draw_histogram
FILENAME="flower.jpg"

# ripped straight from the PNG format specification at 
# https://www.w3.org/TR/PNG-Filters.html
# had to use this due to the inter-dependency of
# pixels in the decoded output.
def paeth_serial_single(left, upper, upper_left):
    initial = left + upper - upper_left
    diff_left = abs(initial - left)
    diff_upper = abs(initial - upper)
    diff_upper_left = abs(initial - upper_left)
    if diff_left <= diff_upper and diff_left <= diff_upper_left:
        return left
    if diff_upper <= diff_upper_left:
        return upper
    return upper_left

def paeth_encode(n, m, img: np.ndarray, output: np.ndarray) -> np.ndarray:
    padded_img = np.pad(img, ((1,0), (1,0)))
    for i in range(n):
        for j in range(m):
            output[i][j] = padded_img[i+1][j+1] - \
                paeth_serial_single(
                    padded_img[i][j+1],
                    padded_img[i+1][j],
                    padded_img[i][j]
                )
    return output

# need to do modulo 256 because the default type cast works differently
# - end up with trimmed values rather than modulated ones.
def paeth_decode(n, m, img: np.ndarray, output: np.ndarray) -> np.ndarray:
    padded_out = np.pad(output, ((1,0), (1,0)))
    for i in range(n):
        for j in range(m):
            padded_out[i+1][j+1] = (img[i][j] + \
                paeth_serial_single(
                    padded_out[i][j+1],
                    padded_out[i+1][j],
                    padded_out[i][j]
                )) % 256
    return padded_out[1:,1:]

# decoding has an output-output dependency so it can't
# be expressed as a numpy matrix operation. Well, unless
# you want to repeat it N times
def paeth_serial(img: np.ndarray, decode=False) -> np.ndarray:
    n, m = img.shape
    output = np.zeros(img.shape)
    if decode:
        res = paeth_decode(n, m, img.astype(np.integer), output.astype(np.integer))
    else:
        res = paeth_encode(n, m, img.astype(np.integer), output.astype(np.integer))
    return res.astype(np.uint8)


if __name__ == "__main__":
    path = Path(argv[1] if len(argv) > 1 else FILENAME)
    img = cv2.imread(path.as_posix(), flags=cv2.IMREAD_GRAYSCALE)
    pred = paeth_serial(img)
    dec = paeth_serial(pred, decode=True)
    cv2.imshow("original in grayscale", img)
    cv2.imshow("serial enc in grayscale", pred + 127)
    cv2.imshow("serial dec in grayscale", dec)
    cv2.waitKey(1000)
    draw_histogram(img, pred + 127)
    cv2.destroyAllWindows()