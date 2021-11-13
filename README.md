# AA_MD
Homework assignments in image processing. Written in Python 3.8.5+, using opencv, matplotlib and numpy

## Histogram
Apply histogram smoothing to an 8-bit grayscale version of the image. Default image - flower.jpg, inclued. Saves the grayscale and smoothed versions of the image in bitmap format, displays the respective histograms.

Implements the histogram equalization algorithm described on wikipedia, using array operations in numpy rather than iteration in native python mainly because that's something I want to practice.

Usage:

```
$ python3 hist.py [optional filename, format that can be parsed by opencv]
```
