import sys

import numpy as np


def check_gradient(grad):
    assert len(grad.shape) == 3 and grad.shape[2] == 3, \
        "Gradient should be a tensor with 3 layers."


def gradient(grid, length_x, length_y=None):
    """
    Calculate the numerical gradient of a matrix in X, Y and Z directions.

    :param grid: Matrix
    :param length_x: Length between two columns
    :param length_y: Length between two rows
    :return:
    """
    if length_y is None:
        length_y = length_x

    assert len(grid.shape) == 2, "Grid should be a matrix."

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

    area = np.sqrt(
        grad[:, :, 0] ** 2 +
        grad[:, :, 1] ** 2 +
        grad[:, :, 2] ** 2
    )
    for i in range(3):
        grad[:, :, i] /= area
    return grad


def aspect(grad, degrees=False):
    """
    Calculate the elevation aspect angle given the gradient.

    Aspect is the direction a slope is facing to.

    :param grad: Tensor representing the X,Y,Z gradient
    :param degrees: Output in degrees or radians
    :return: Matrix with aspect per grid cell.
    """
    check_gradient(grad)

    y_grad = grad[:, :, 1]
    x_grad = grad[:, :, 0]
    asp = np.arctan2(y_grad, x_grad) + (np.pi / 2)
    asp[asp < 0] += (2 * np.pi)

    if degrees:
        asp = np.rad2deg(asp)
    return asp


def slope(grad, degrees=False):
    """
    Calculate the slope inclination angle given the gradient.
    :param grad: Tensor representing the X,Y,Z gradient
    :param degrees:
    :return:
    """
    check_gradient(grad)

    sl = np.arccos(grad[:, :, 2])
    if degrees:
        sl = np.rad2deg(sl)
    return sl


# TODO TdR 26/08/16: test
def normal_vector(slope, aspect):
    """
    Calculate the unit vector normal to the surface defined by slope and aspect.
    :param slope: slope inclination in degrees
    :param aspect: slope aspect in degrees
    :return: 3-dim unit normal vector
    """
    slope_rad = np.deg2rad(slope)
    aspect_rad = np.deg2rad(aspect)

    nvx = np.sin(aspect_rad) * np.sin(slope_rad)
    nvy = -np.cos(aspect_rad) * np.sin(slope_rad)
    nvz = np.cos(slope_rad)
    return np.array([nvx, nvy, nvz])


def hill_shade(gradient, sun_vector):
    """
    Compute the intensity of illumination on a surface given the sun position.
    :param gradient:
    :param sun_vector:
    :return:
    """
    check_gradient(gradient)

    hsh = (
        gradient[:, :, 0] * sun_vector[0] +
        gradient[:, :, 1] * sun_vector[1] +
        gradient[:, :, 2] * sun_vector[2]
    )
    # Remove negative incidence angles - indicators for self-shading.
    hsh = (hsh + abs(hsh)) / 2.

    return hsh


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


def raycast(dem, sun_vector, dl):
    """Cast shadows on the DEM from a given sun position."""

    inverse_sun_vector = _invert_sun_vector(sun_vector)
    normal_sun_vector = _normalize_sun_vector(sun_vector)

    rows, cols = dem.shape
    z = dem.T

    # Determine sun direction.
    if sun_vector[0] < 0:
        # The sun shines from the West.
        start_col = 1
    else:
        # THe sun shines from the East.
        start_col = cols - 1

    if sun_vector[1] < 0:
        # The sun shines from the North.
        start_row = 1
    else:
        # The sun shines from the South.
        start_row = rows - 1

    in_sun = np.ones_like(z)
    # Project West-East
    row = start_row
    for col in range(cols):
        _cast_shadow(row, col, rows, cols, dl, in_sun, inverse_sun_vector,
                     normal_sun_vector, z)

    # Project North-South
    col = start_col
    for row in range(rows):
        _cast_shadow(row, col, rows, cols, dl, in_sun, inverse_sun_vector,
                     normal_sun_vector, z)
    return in_sun.T


def _normalize_sun_vector(sun_vector):
    normal_sun_vector = np.zeros(3)
    normal_sun_vector[2] = np.sqrt(sun_vector[0] ** 2 + sun_vector[1] ** 2)
    normal_sun_vector[0] = -sun_vector[0] * sun_vector[2] / normal_sun_vector[2]
    normal_sun_vector[1] = -sun_vector[1] * sun_vector[2] / normal_sun_vector[2]
    return normal_sun_vector


def _invert_sun_vector(sun_vector):
    return -sun_vector / max(abs(sun_vector[:2]))


def _cast_shadow(row, col, rows, cols, dl, in_sun, inverse_sun_vector,
                 normal_sun_vector, z):
    n = 0
    z_previous = -sys.float_info.max
    while True:
        # Calculate projection offset
        dx = inverse_sun_vector[0] * n
        dy = inverse_sun_vector[1] * n
        col_dx = int(round(col + dx))
        row_dy = int(round(row + dy))
        if (col_dx < 0) or (col_dx >= cols) or (row_dy < 0) or (row_dy >= rows):
            break

        vector_to_origin = np.zeros(3)
        vector_to_origin[0] = dx * dl
        vector_to_origin[1] = dy * dl
        vector_to_origin[2] = z[col_dx, row_dy]
        z_projection = np.dot(vector_to_origin, normal_sun_vector)

        if z_projection < z_previous:
            in_sun[col_dx, row_dy] = 0
        else:
            z_previous = z_projection
        n += 1