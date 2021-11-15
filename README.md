# AA_MD
Homework assignments in image processing. Written in Python 3.8.5+, using opencv-python, matplotlib and numpy. Developed on linux but tested on Windows 10, with:
- Python 3.8.10
- opencv-python 4.5.4.58
- matplotlib 3.4.3
- numpy 1.21.4

Working with color opens up a whole lot of new problems due to the complexity of conserving proper mappings between the various representation spaces so everything in this repository works on grayscale versions of the input image.

## Histogram
Apply histogram smoothing to an 8-bit grayscale version of the image. Default image - flower.jpg, inclued. Saves the grayscale and smoothed versions of the image in bitmap format in the histogram/ directory, displays both images and their respective histograms. Close the histogram window to terminate execution.

Implements the histogram equalization algorithm described on wikipedia, using array operations in numpy rather than iteration in native python mainly because that's something I want to practice.

Usage:

```
$ python3 hist.py
$ python3 hist.py flower.jpg
```
## Gauss
Gaussian blur on a grayscale version of the image. Default image - flower.jpg, inclued. Saves the grayscale and blurred versions of the image in bitmap format in the gauss/ directory, displays both images. Press any key to terminate execution.

Takes optional arguments - filename, kernel radius, "fast". When running with the "fast" argument, OpenCV's implementation of kernel convolution is used, which is 1000+ times faster for the example image. Otherwise the image is iterated over in python to show that I understand how convolution is supposed to work. The kernel is generated from the radius parameter, with a standard deviation of radius / 3. Edges are handled by normalization - the image is padded with 0-valued pixels and a normalization constant is computed to make sure the kernel cells always add up to the same quantity when overlapping the edges of the image.

Usage

```
$ python3 gauss.py
$ python3 gauss.py flower.jpg
$ python3 gauss.py flower.jpg fast
$ python3 gauss.py flower.jpg 15
$ python3 gauss.py flower.jpg 15 fast
```


## Fourier
Fourier transform and low pass filter on a grayscale version of the image. Default image - flower.jpg, inclued. Saves the grayscale version, fourier transform, filtered transform, inverse of the transform and inverse of the filtered transform in the fourier/ directory. Press any key to terminate execution.

Takes optional arguments - filename, filter radius, "fast". When running with the "fast" argument, OpenCV's implementation of FFT is used, which is around 30 times faster for the example image. Otherwise a straightforward implementation of the DFT algorithm (using matrix multiplication) written by me is used. The lowpass filter is a simple radial distance threshold mask applied to the transform.

Usage

```
$ python3 fourier.py
$ python3 fourier.py flower.jpg
$ python3 fourier.py flower.jpg fast
$ python3 fourier.py flower.jpg 40
$ python3 fourier.py flower.jpg 40 fast
```

## Bilinear
Upscale a grayscale version of the image using bilinear interpolation. Default image - flower.jpg, inclued. Saves the grayscale version and upscaled in the bilinear/ directory. Press any key to terminate execution.

Takes optional arguments - filename, scale. Scale needs to be an integer and greater than 1, the default value is 4. 

Usage

```
$ python3 bilinear.py
$ python3 bilinear.py flower.jpg
$ python3 bilinear.py flower.jpg 2
```
