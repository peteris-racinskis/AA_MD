import cv2
import numpy as np
from pathlib import Path
from sys import argv
from typing import List, Tuple
from hist import draw_histogram
FILENAME="flower.jpg"

# zero out the previous values for the first row/column as necessary
def shifted(img: np.ndarray, shifts: List[Tuple]) -> List[np.ndarray]:
    result = []
    for shift in shifts:
        rolled = np.roll(img, shift, (0,1))
        rolled[0,:] = 0 if shift[0] == 1 else rolled[0,:]
        rolled[:,0] = 0 if shift[1] == 1 else rolled[:,0]
        result.append(rolled)
    return result

# if you allow 8-bit overflow in this function it doesn't work. cast to system
# wordlength.
def diff(base: np.ndarray, values: List[np.ndarray]) -> np.ndarray:
    result = []
    for value in values:
        result.append(np.abs(value.astype(np.integer) - base.astype(np.integer)))
    return np.stack(result).astype(np.uint8)


# This rolls over the edge of the image,
# but for large images that basically doesn't matter
# (I hope)
def predicted(img: np.ndarray) -> np.ndarray:
    upper, left, upper_left = shifted(img, [(1,0), (0,1), (1,1)])
    initial = upper + left - upper_left
    stacked = np.stack([upper, left, upper_left])
    diffs = diff(initial, [upper, left, upper_left])
    max_index = np.argmin(diffs, axis=0)
    x_grid, y_grid = np.meshgrid(np.arange(img.shape[0]),
     np.arange(img.shape[1]), indexing="ij")
    prediction = stacked[max_index, x_grid, y_grid]
    return prediction

def deviation(img: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return (img - pred)


if __name__ == "__main__":
    path = Path(argv[1] if len(argv) > 1 else FILENAME)
    img = cv2.imread(path.as_posix(), flags=cv2.IMREAD_GRAYSCALE)
    pred = predicted(img)
    d = deviation(img, pred)
    cv2.imshow("original in grayscale", img)
    cv2.imshow("predicted in grayscale", pred)
    cv2.imshow("deviation in grayscale", d)
    cv2.waitKey(1000)
    draw_histogram(img, d)
    cv2.destroyAllWindows()