import cv2
import numpy as np
from pathlib import Path
from sys import argv
from math import cos, sin, radians, sqrt
from thinning import force_binary
from fourier import pixels
from typing import List, Tuple, AbstractSet
FILENAME="hough-input-interrupted.png"
INF=99999999999
DRAW_DELAY=0
THRESHOLD=150
CLUSTER_HALT_DIST=5
NUM_CLUSTERS=1
RESOLUTION=400
LO=0
HI=255
DC=127
eps = 0.5

# a very sub-otpimal way of doing this but saves tons of 
# debugging time - just solve a linear system of equations 
# for each theta and point to find r rather than trying to
# come up with a general geometric rule for doing this.
def radius(x1, x2, theta):
    r1, r2 = cos(theta), sin(theta)
    coefficients = np.asarray([
        [r1, 1, 0],
        [r2, 0, 1],
        [0, r1, r2]
    ])
    ordinates = np.asarray([
        x1,
        x2,
        0
    ])
    k, _, __ = np.linalg.solve(coefficients, ordinates)
    return int(k)

# again, slow but it works.
def discretize(value, limit, bins):
    return int(min((value * bins) / limit, bins-1))

def restore(value, limit, bins):
    return value * limit / bins

# the same line can be represented with r, theta and -r, theta + pi
# so while the (0,...,360) transform is interesting to look at, it's
# best to look at just half of it to find lines - each line will be
# represented by 2 clusters otherwise.
def hough_transform(img: np.ndarray, resolution=RESOLUTION, full=False) -> np.ndarray:
    n,m = img.shape
    r_limit = int(sqrt(n ** 2 + m ** 2)) + 1
    t_limit = 360 if full else 180
    hough_space = np.zeros((resolution, resolution))
    for i in range(n):
        for j in range(m):
            if img[i][j] == HI:
                t_range = np.linspace(0, t_limit, resolution)
                for t in t_range:
                    r = radius(i,j,radians(t))
                    r_bin = int(discretize(r, r_limit, resolution / 2) + resolution / 2)
                    t_bin = discretize(t, t_limit, resolution)
                    hough_space[r_bin][t_bin] += 1
    return hough_space, r_limit, resolution

def normalize_to_8bit(img: np.ndarray) -> np.ndarray:
    scaler = 255 / np.max(img)
    return (img * scaler).astype(np.uint8)

def threshold(img: np.ndarray, thresh) -> np.ndarray:
    return np.where(img > thresh, 255, 0).astype(np.uint8)

# use the min euclidean distance metric
def cluster_distance(c1: set, c2: set) -> float:
    dmin = INF
    for x1, y1 in c1:
        for x2, y2 in c2:
            d = sqrt((x1-x2)**2 + (y1-y2)**2)
            if d < dmin:
                dmin = d
    return dmin

# lazy, inefficient implementation of the well known hierarchical
# clustering algorithm. No point in optimizing this since it's
# not a bottleneck.
def hierarchical_cluster(img: np.ndarray) -> List[AbstractSet[Tuple]]:
    n, m = img.shape
    clusters = []
    for i in range(n):
        for j in range(m):
            if img[i][j] == HI:
                clusters.append(set([(i,j)]))
    while len(clusters) > NUM_CLUSTERS:
        distances = dict()
        for cluster in clusters:
            for remaining in clusters:
                if cluster == remaining:
                    continue
                distance = cluster_distance(cluster, remaining)
                distances[distance] = (cluster, remaining)
        mindist = min(distances.keys())
        if mindist > CLUSTER_HALT_DIST:
            break
        A, B = distances[mindist]
        AB = A.union(B)
        clusters.remove(A)
        clusters.remove(B)
        clusters.append(AB)
    return clusters

def cluster_centers(clusters: List[AbstractSet[Tuple]]) -> List[Tuple]:
    centers = []
    for cluster in clusters:
        n = len(cluster)
        xsum = 0
        ysum = 0
        for x,y in cluster:
            xsum += x
            ysum += y
        centers.append((xsum/n,ysum/n)) # this line was wrong and took me about 2 hours to find!
    return centers

# really ugly but it does the trick - restore the original radius/theta,
# then create two inequality masks - drawing a line between them in the 
# deadzone defined by +- epsilon. Wouldn't work with exact equalities
def draw_line(img: np.ndarray, r_b, t_b, r_limit, bins):
    N, M = img.shape
    theta = radians(restore(t_b, 180, bins))
    r = restore(r_b - bins / 2, r_limit, bins / 2)
    y_grid, x_grid = np.meshgrid(np.arange(M), np.arange(N))
    free = img == 0
    greater = (x_grid * np.cos(theta) + y_grid * np.sin(theta)) >= r - eps
    lesser = (x_grid * np.cos(theta) + y_grid * np.sin(theta)) <= r + eps
    mask = greater & lesser & free
    result = np.where(mask, DC, img)
    return result.astype(np.uint8)


if __name__ == "__main__":
    path = Path(argv[1] if len(argv) > 1 else FILENAME)
    img = force_binary(cv2.imread(path.as_posix(), flags=cv2.IMREAD_GRAYSCALE))
    transform, rlim, bins = hough_transform(img)
    transform = normalize_to_8bit(transform)
    mirrored, _, __ = hough_transform(img, full=True)
    mirrored = normalize_to_8bit(mirrored)
    thresholded = threshold(transform, THRESHOLD)
    clusters = hierarchical_cluster(thresholded)
    centers = cluster_centers(clusters) 
    drawn_lines = img
    for r, theta in centers:
        drawn_lines = draw_line(drawn_lines, r, theta, rlim, bins)

    cv2.imwrite("hough/{}_original-forced-binary.bmp".format(path.stem), img)
    cv2.imwrite("hough/{}_transform-180.bmp".format(path.stem), transform)
    cv2.imwrite("hough/{}_transform-360.bmp".format(path.stem), mirrored)
    cv2.imwrite("hough/{}_thresholded.bmp".format(path.stem), thresholded)
    cv2.imwrite("hough/{}_inferred-lines.bmp".format(path.stem), drawn_lines)

    cv2.imshow("original binary image", img)
    cv2.imshow("hough transform", transform)
    cv2.imshow("hough transform full", mirrored)
    cv2.imshow("thresholded for clustering", thresholded)
    cv2.imshow("lines from cluster centers", drawn_lines)
    cv2.waitKey(DRAW_DELAY)
    cv2.destroyAllWindows()
