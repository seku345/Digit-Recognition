import numpy as np
from PIL import Image
from scipy.ndimage import measurements, interpolation


def center_image(img):
    cy, cx = measurements.center_of_mass(img)
    rows, cols = img.shape

    shiftX = np.round(cols/2.0-cx).astype(int)
    shiftY = np.round(rows/2.0-cy).astype(int)

    return interpolation.shift(img, [shiftY, shiftX])


def get_pixels(path: str) -> np.array:
    image = Image.open('number.png').convert('L')
    image = image.resize((28, 28))

    img_data = np.asarray(image)

    img_data_centered = center_image(np.asarray(img_data))

    numpy_data = img_data_centered.flatten()

    Image.fromarray(img_data_centered).save('edited_number.png')

    return numpy_data
