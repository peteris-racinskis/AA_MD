import cv2
import numpy as np
from pathlib import Path
from sys import argv, float_info
from math import radians, degrees, cos, acos, sqrt, pi
FILENAME="flower.jpg"
DRAW_DELAY=300
ANGLE=90
eps = float_info.epsilon

def rgb_to_hsi(img: np.ndarray) -> np.ndarray:
    B, G, R = [np.squeeze(x) for x in np.split(img, 3, axis=2)]
    b = B / (R + G + B + eps)
    g = G / (R + G + B + eps)
    r = R / (R + G + B + eps)
    n, m = b.shape
    h = np.zeros((n,m))
    s = np.zeros((n,m))
    for j in range(n):
        for k in range(m):
            b_t = b[j][k]
            g_t = g[j][k]
            r_t = r[j][k]
            h_t = acos(
                0.5 * (r_t - g_t + r_t - b_t) /
                (sqrt(
                    (r_t - g_t) ** 2 +
                    (r_t - b_t) * (g_t - b_t)
                ) + eps)
            )
            if b_t <= g_t:
                h[j][k] = h_t
            else:
                h[j][k] = (2 * pi) - h_t
            s[j][k] = 1 - (3 * min(b_t, g_t, r_t))
    i = (R + G + B) / (3 * 255)
    res = np.dstack([h,s,i])
    return res

# to check what order the colors are in
def leave_one(img: np.ndarray, leave=3) -> np.ndarray:
    b, g, r = [np.squeeze(x) for x in np.split(img, 3, axis=2)]
    b = np.zeros(b.shape) if not leave == 1 else b
    g = np.zeros(g.shape) if not leave == 2 else g
    r = np.zeros(r.shape) if not leave == 3 else r
    res = np.dstack([b,g,r])
    return res.astype(np.uint8)


def proper_angle(radians):
    return degrees(radians) % 360

def hsi_to_rgb(img: np.ndarray) -> np.ndarray:
    h, s, i = [np.squeeze(x) for x in np.split(img, 3, axis=2)]
    n, m = h.shape
    x = i * (1-s)
    y = np.zeros((n,m))
    for j in range(n):
        for k in range(m):
            h_t = h[j][k]
            i_t = i[j][k]
            s_t = s[j][k]
            prop = proper_angle(h_t)
            if prop <= 120:
                #alpha = 0
                #beta = radians(60)
                h_t = h_t - radians(0)
            elif prop <= 240:
                #alpha = radians(120)
                #beta = radians(180)
                h_t = h_t - radians(120)
            else:
                #alpha = radians(240)
                #beta = radians(300)
                h_t = h_t - radians(240)
            #y[j][k] = i_t * ( 1 + (s_t * cos(h_t - alpha) / cos(beta - h_t)))
            y[j][k] = i_t * ( 1 + (s_t * cos(h_t) / cos(radians(60) - h_t)))
            #beta = pi / 3
            #y[j][k] = i_t * ( 1 + (s_t * cos(h_t) / cos(beta - h_t)))
    #y = i * (1 + (s * np.cos(h)) / (np.cos(np.pi / 3 - h)))
    z = 3 * i - (x + y)
    b = np.zeros((n,m))
    r = np.zeros((n,m))
    g = np.zeros((n,m))
    for j in range(n):
        for k in range(m):
            x_t = x[j][k]
            y_t = y[j][k]
            z_t = z[j][k]
            prop = proper_angle(h[j][k])
            if prop <= 120:
                l = [x_t, z_t, y_t]
            elif prop <= 240:
                l = [z_t, y_t, x_t]
            else:
                l = [y_t, x_t, z_t]
            b[j][k], g[j][k], r[j][k] = l
    res = np.dstack([b,g,r])
    res = res * 255
    res = res % 255
    return res.astype(np.uint8)



def hue_rotate(img: np.ndarray, degrees) -> np.ndarray:
    r = radians(degrees)
    h, s, i = [np.squeeze(x) for x in np.split(img, 3, axis=2)]
    h = h + r
    rotated = np.dstack([h,s,i])
    return rotated



if __name__ == "__main__":
    path = Path(argv[1] if len(argv) > 1 else FILENAME)
    img = cv2.imread(path.as_posix(), flags=cv2.IMREAD_COLOR)
    angle = int(argv[2]) if len(argv) > 2 and argv[2].isdigit() else ANGLE

    hsi = rgb_to_hsi(img)
    bgr = hsi_to_rgb(hsi)
    rot = hue_rotate(hsi, angle)

    b = leave_one(img, 1)
    g = leave_one(img, 2)
    r = leave_one(img, 3)
    cv2.imshow("blue", b)
    cv2.imshow("green", g)
    cv2.imshow("red", r)
    cv2.imshow("rev", bgr)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()