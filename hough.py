import cv2
import numpy as np
from pathlib import Path
from sys import argv
FILENAME="hough-input.jpg"
LO=0
HI=255
DC=127


if __name__ == "__main__":
    path = Path(argv[1] if len(argv) > 1 else FILENAME)