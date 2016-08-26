import numpy as np


def gradient(grid, length_x, length_y=None):
    if length_y is None:
        length_y = length_x

    # grid is a matrix of size (x, y)
    assert len(grid.shape) == 2

    grad = np.empty((*grid.shape, 3))
    grad[:] = np.nan
    grad[:-1, :-1, 0] = 0.5 * length_y * (
        grid[:-1, :-1] - grid[:-1, 1:] + grid[1:, :-1] - grid[1:, 1:]
    )
    grad[:-1, :-1, 1] = 0.5 * length_x * (
        grid[:-1, :-1] + grid[:-1, 1:] - grid[1:, :-1] - grid[1:, 1:]
    )
    grad[:-1, :-1, 2] = length_x * length_y

    # Copy last row and column
    grad[-1, :, :] = grad[-2, :, :]
    grad[:, -1, :] = grad[:, -2, :]

    # TODO TdR 26/08/16: use vector calculation
    area = np.sqrt(
        grad[:, :, 0] ** 2 +
        grad[:, :, 1] ** 2 +
        grad[:, :, 2] ** 2
    )
    for i in range(3):
        grad[:, :, i] /= area
    return grad


def aspect(grad, degrees=False):
    assert len(grad.shape) == 3 and grad.shape[2] == 3

    y_grad = grad[:, :, 1]
    x_grad = grad[:, :, 0]
    asp = np.arctan2(y_grad, x_grad) + (np.pi / 2)
    asp[asp < 0] += (2 * np.pi)

    if degrees:
        asp = np.rad2deg(asp)
    return asp


def slope(grad, degrees=False):
    # TODO TdR 25/08/16: Always gives back constant values. Check if correct.
    sl = np.arccos(grad[:, :, 2])
    if degrees:
        sl = np.rad2deg(sl)
    return sl


# TODO TdR 26/08/16: test
def normal_vector(slope, aspect):
    slope_rad = np.deg2rad(slope)
    aspect_rad = np.deg2rad(aspect)

    nvx = np.sin(aspect_rad) * np.sin(slope_rad)
    nvy = -np.cos(aspect_rad) * np.sin(slope_rad)
    nvz = np.cos(slope_rad)
    return np.array([nvx, nvy, nvz])


def hill_shade(gradient, sun_vector):
    # TODO TdR 25/08/16: turn into np vectorized multiplication.
    hsh = (
        gradient[:, :, 0] * sun_vector[0] +
        gradient[:, :, 1] * sun_vector[1] +
        gradient[:, :, 2] * sun_vector[2]
    )
    hsh = (hsh + abs(hsh)) / 2.
    return hsh


def shade():
    # From doshade
    pass




def sun_position():
    pass

def sun_vector():
    pass

def declination():
    pass

def insolation():
    pass

def day_length():
    pass
