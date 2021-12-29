# AA_MD
Homework assignments in image processing. Written in Python 3.8.5+, using opencv-python, matplotlib and numpy. Developed on linux but tested on Windows 10, with:
- Python 3.8.10
- opencv-python 4.5.4.58
- matplotlib 3.4.3
- numpy 1.21.4

Working with color creates a whole lot of new problems due to the complexity of conserving proper mappings between the various representation spaces so everything in this repository works on grayscale versions of the input image unless otherwise specified.

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

NOTE: (12/12/21) fixed roll indexing to use ceiling division by 2 rather than floor. This is correct for even numbered axis lengths and has a shift error of 0.5 rather than 1.5 for odd ones.

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

## Wavelet
Produces a Discrete Wavelet Transform of an 8-bit grayscale version of the image using Haar wavelets, applies hard and soft thresholding to the transform for denoising. Looking at the output images it is obvious that hard thresholding is not a feasible approach due to ringing, which is much reduced in the soft threshold case. For a practical implementation, a more complex thresholding approach should be used but soft thresholding was deemed sufficient for this homework application.

Outputs a grayscale version of the original, the DWT normalized for small magnitude components to be visible, an inverse DWT without any thresholding and inverse DWTs with hard and soft thresholding to the wavelet/ directory. Optional argument - input file name. By default takes a version of the flower.jpg image with some RGB noise applied in GIMP - flower-noise.jpg.

Press any key with any of the opencv windows selected to terminate program execution.

Usage

```
$ python3 wavelet.py
$ python3 wavelet.py flower-noise.jpg
```

## Predictive
Compress an 8-bit grayscale image with predictive coding. Prediction done using the Paeth predictor. Compression done with zlib. Does both lossless and lossy predictive coding with a threshold parameter (default = 10). Outputs a grayscale version of the original, the decompressed predictive codings and the decompressed, decoded final images and their difference to the predictive/ directory. Draws a histogram of the losless and lossy predictive codings. Console output - size of output when compressing respectively: the original image, the lossless predictive encoding, the lossy predictive encoding.

Takes an optional argument - filename. Close the histogram window to terminate program execution. If images aren't geting drawn, increase the DRAW_DELAY constant as it might be system specific.

Usage

```
$ python3 predictive.py
$ python3 predictive.py flower.jpg
```

Output when running with default parameters on my system

```
Size of directly compressed original:   30374
Size of compressed predictive w/o loss: 21095
Size of compressed predictive w/t loss: 7689
```

## Edge-kernel
Implement convolution for an 8-bit grayscale version of the image with the Laplace and Sobel kernels for edge detection purposes. Apply Gaussian blur to the original image before edge detection to illustrate the differences between the original and blurred version. Outputs the grayscale original, blurred version, both outputs after convolution with the Laplace kernel and all outputs after convolution with the Sobel kernels - horizontal, vertical and magnitude sum of both components, for each source image - to the edge-kernel/ directory.

Default input file - flower.jpg. Optional filename argument. Press any key with any of the opencv windows selected to terminate program execution.

Usage

```
$ python3 erge-kernel.py
$ python3 edge-kernel.py flower.jpg
```

## Color


## Hough