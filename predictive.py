import cv2
import numpy as np
from pathlib import Path
from sys import argv
from typing import List, ByteString
from hist import draw_histogram
import zlib
FILENAME="flower.jpg"
SIZE_BYTES=4
SIZE_ORDER='big'

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

# this is a somewhat clumsy implementation but it appears to work
def paeth_encode(n, m, img: np.ndarray, output: np.ndarray, thresh) -> np.ndarray:
    padded_img = np.pad(img, ((1,0), (1,0)))
    for i in range(n):
        for j in range(m):
            output[i][j] = padded_img[i+1][j+1] - \
                paeth_serial_single(
                    padded_img[i][j+1],
                    padded_img[i+1][j],
                    padded_img[i][j]
                )
            # In the lossy case, replace output pixel with 0 and
            # replace the input pixel with whatever the result of
            # the decode operation would be at this location
            if thresh > 0:
                if abs(output[i][j]) < thresh:
                    output[i][j] = 0
                    padded_img[i+1][j+1] = (output[i][j] + \
                        paeth_serial_single(
                            padded_img[i][j+1],
                            padded_img[i+1][j],
                            padded_img[i][j]
                        )) % 256
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
def paeth_serial(img: np.ndarray, decode=False, thresh=0) -> np.ndarray:
    n, m = img.shape
    output = np.zeros(img.shape)
    if decode:
        res = paeth_decode(n, m, img.astype(np.int64), output.astype(np.int64))
    else:
        res = paeth_encode(n, m, img.astype(np.int64), output.astype(np.int64), thresh)
    return res.astype(np.uint8)

def compress(input: np.ndarray) -> ByteString:
    shape = b''.join([x.to_bytes(SIZE_BYTES, SIZE_ORDER) for x in input.shape])
    data = input.tobytes()
    return zlib.compress(shape+data, level=9)

def decompress(input: ByteString) -> np.ndarray:
    byte_array = zlib.decompress(input)
    shape = (int.from_bytes(byte_array[:SIZE_BYTES], SIZE_ORDER),
                int.from_bytes(byte_array[SIZE_BYTES:2*SIZE_BYTES], SIZE_ORDER))
    arr = np.frombuffer(byte_array[2*SIZE_BYTES:], dtype=np.uint8)
    return np.reshape(arr, shape)


if __name__ == "__main__":
    path = Path(argv[1] if len(argv) > 1 else FILENAME)
    img = cv2.imread(path.as_posix(), flags=cv2.IMREAD_GRAYSCALE)
    pred_noloss = paeth_serial(img)
    pred_wtloss = paeth_serial(img, thresh=10)
    c_origin = compress(img)
    c_noloss = compress(pred_noloss)
    c_wtloss = compress(pred_wtloss)
    dc_noloss = decompress(c_noloss)
    dc_wtloss = decompress(c_noloss)
    dec_noloss = paeth_serial(dc_noloss, decode=True)
    dec_wtloss = paeth_serial(dc_wtloss, decode=True)

    print(f"Size of directly compressed original:\t{len(c_origin)}")
    print(f"Size of compressed predictive w/o loss:\t{len(c_noloss)}")
    print(f"Size of compressed predictive w/t loss:\t{len(c_wtloss)}")

    cv2.imwrite("predictive/{}_original.bmp".format(path.stem), img)
    cv2.imwrite("predictive/{}_enc_noloss.bmp".format(path.stem), dc_noloss)
    cv2.imwrite("predictive/{}_enc_wtloss.bmp".format(path.stem), dc_wtloss)
    cv2.imwrite("predictive/{}_dec_noloss.bmp".format(path.stem), dec_noloss)
    cv2.imwrite("predictive/{}_dec_wtloss.bmp".format(path.stem), dec_wtloss)

    cv2.imshow("original in grayscale", img)
    cv2.imshow("serial enc in grayscale", dc_noloss + 127)
    cv2.imshow("serial enc w loss in grayscale", dc_wtloss + 127)
    cv2.imshow("serial dec in grayscale", dec_noloss)
    cv2.imshow("serial dec w loss in grayscale", dec_wtloss)
    cv2.waitKey(1000)
    draw_histogram(pred_noloss + 127, pred_wtloss + 127)
    cv2.destroyAllWindows()