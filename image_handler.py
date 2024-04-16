import numpy as np
from PIL import Image


def get_pixels(path: str) -> np.array:
    image = Image.open('number.png').convert('L')

    numpyData = np.asarray(image).flatten()

    return numpyData
