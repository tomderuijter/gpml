import datetime as dt

import numpy as np

import altitude
import gpml.area as area
import python_dem_shadows.python_dem_shadows as dem


def get_elevations(points):
    e = altitude.ElevationService('.cache/')

    elevations = []
    last_elevation = 0
    for point in points:
        elevation = e.get_elevation(*point)

        if elevation is None:
            elevation = last_elevation
        else:
            last_elevation = elevation
        elevations.append(elevation)
    return np.array(elevations)


def blur(mat, radius=1):
    from scipy import ndimage as ndi
    bl = ndi.gaussian_filter(mat, radius)
    return bl


def get_gradient_features(request, elevations):
    lat_len, lon_len = area.calculate_request_lengths(request)
    gradients = dem.gradient(elevations, lon_len, lat_len)
    aspect = dem.aspect(gradients, degrees=False)
    slope = dem.slope(gradients, degrees=False)
    return gradients, aspect, slope


def get_shade_features(request, gradients):
    sv = calculate_sun_vector(request)
    hsh = dem.hill_shade(gradients, sv)
    return hsh


def get_shadow_features(request, elevations):
    sv = calculate_sun_vector(request)
    lat_len, lon_len = area.calculate_request_lengths(request)
    sh = dem.project_shadows(elevations, sv, lon_len, lat_len)
    return sh


def calculate_sun_vector(request):
    timezone = 1
    center_lat = (request['predict_area'][1] + request['predict_area'][0]) / 2
    center_lon = (request['predict_area'][3] + request['predict_area'][2]) / 2
    valid_time = request['start'] + dt.timedelta(
        hours=request['forecast_hours'][0])
    sv = dem.sun_vector(dem.to_juliandate(valid_time), center_lat, center_lon,
                        timezone)
    return sv
