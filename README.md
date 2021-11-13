# AA_MD
Homework assignments in image processing. Written in Python 3.8.5+, using opencv, matplotlib and numpy

## Histogram
Apply histogram smoothing to an 8-bit grayscale version of the image. Default image - flower.jpg, inclued. Saves the grayscale and smoothed versions of the image in bitmap format, displays both images and their respective histograms. Close the histogram window to terminate execution.

Implements the histogram equalization algorithm described on wikipedia, using array operations in numpy rather than iteration in native python mainly because that's something I want to practice.

Usage:

```
$ python3 hist.py
$ python3 hist.py flower.jpg
```
## Gauss
Gaussian blur on a grayscale version of the image. Default image - flower.jpg, inclued. Saves the grayscale and blurred versions of the image in bitmap format, displays both images. Press any key to terminate execution.

Takes optional arguments - filename, kernel radius, "fast". When running with the "fast" argument, OpenCV's implementation of kernel convolution is used, which is 1000+ times faster. Otherwise the image is iterated over in python to show that I understand how convolution is supposed to work. The kernel is generated from the radius parameter, with a standard deviation of radius / 3. Edges are handled by normalization - the image is padded with 0-valued pixels and a normalization constant is computed to make sure the kernel cells always add up to the same quantity when overlapping the edges of the image.

Usage

```
$ python3 gauss.py
$ python3 gauss.py flower.jpg
$ python3 gauss.py flower.jpg fast
$ python3 gauss.py flower.jpg 15
$ python3 gauss.py flower.jpg 15 fast
```