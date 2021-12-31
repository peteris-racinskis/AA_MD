# AA_MD
Homework assignments in image processing. Written in Python 3.8.5+, using opencv-python, matplotlib and numpy. Developed on linux but tested on Windows 10, with:
- Python 3.8.10
- opencv-python 4.5.4.58
- matplotlib 3.4.3
- numpy 1.21.4

Working with color creates a number of new problems and is a massive pain in debugging terms, so everything in this repository works on grayscale versions of the input image.

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
Implement convolution for an 8-bit grayscale version of the image with the Laplace and Sobel kernels for edge detection purposes. Apply Gaussian blur to the original image before edge detection to illustrate the differences between the original and blurred version. Convolve both versions of the image with each of the kernels, normalize results to illustrate kernel effects. Implement threshold edge detection on the outputs of CV2's Laplacian operator and both the Laplace and Sobel operators written here. Outputs the grayscale original, all normalized convolutions and edge detection results to the edge-kernel/ directory. 

Note: most practical implementations of the Laplacian operator seem to use a Gaussian-Laplacian combined kernel, but a similar effect is obtained here by blurring the image first.

Default input file - flower.jpg. Optional filename argument. Press any key with any of the opencv windows selected to terminate program execution.

Usage

```
$ python3 erge-kernel.py
$ python3 edge-kernel.py flower.jpg
```

## Thinning
Morphological thinning of a binary image. All non-zero values are forced to the maximum pixel value. Outputs the binarized version of the input image as well as the thinned image to the thinned/ directory. Console output logs the number of iterations required for convergence. 

Optional argument - filename, default input file - thin-input.png, (a few random thick curves drawn with the brush tool in GIMP). To terminate program execution, press any key with any of the opencv windows selected.

Usage

```
$ python3 thinning.py
$ python3 thinning.py thin-input.png
```
Output when running with default parameters

```
Converged after 15 iterations
```

## Hough
Computes the hough transform of a binary input image, applies a threshold and uses hierarchical clusterization to find the top few clusters in radius-theta space. Termination conditions for clusterization - 2 clusters remaining or minimum inter-cluster distance greater than some threshold. These can be adjusted in the script itself. Outputs the binarized original, 180-degree view of the transform, 360-degree (mirrored, families of sinusioids converging at two theta values are formed by the same line) view of the transform, thresholded transform and the inferred lines drawn on top of the original image to the hough/ directory, which has been populated with some example outputs. 

This is a very inefficient implementation since the radius for the line at each angle is computed by solving a system of linear equations, and in general no attempt has been made to avoid calling Python functions inside nested loops, so running it on images much larger than the ones provided as examples is discouraged. Computing the example output takes a few seconds on my machine, which is fairly powerful. 

Absolutely no attempt has been made to optimize performance in this case, but it can be improved by reducing the "RESOLUTION" parameter at the top of the script (resulting in fewer angle values being checked at each point). The tradeoff for this is that the inferred lines are less accurate - 200 is good enough for some examples but results in an offset with others, while 400 is an improvement but results in longer runtimes (input-2 suffers from this at 200, whereas the example with input-2-interrupted was done at 400, eliminating the offset).

To illustrate that this algorithm can work with distorted lines, input-3 and input-4 have been provided. With input-3 and NUM_CLUSTERS=2 one can see that a single strong line can dominate a weak one (the image is normalized before thresholding, which means very strong outputs from one line can push those from another below the threshold). With input-4 one can see that (having set NUM_CLUSTERS=1 to produce just 1 line) the weak line (produced by drawing a line by hand and deleting segments) can still be detected by this algorithm.

Optional argument - filename. Default input file - hough-input-interrupted.png. Press any key with any of the opencv windows selected to terminate program execution.

Usage

```
$ python3 hough.py
$ python3 hough.py hough-input-interrupted.png
```

