from math import radians, cos


def calculate_request_lengths(request):
    mean_latitude = (
                        request['predict_area'][1] - request['predict_area'][0]
                    ) * 0.5

    lat_len, lon_len = degree_distance(mean_latitude)
    factor = request['predict_resolution']
    lat_len *= factor
    lon_len *= factor
    return lat_len, lon_len


def degree_distance(latitude):

    lat = radians(latitude)

    # Latitude calculation constants
    m1 = 111132.92
    m2 = -559.82
    m3 = 1.175
    m4 = -0.0023

    # Longitude calculation constants
    p1 = 111412.84
    p2 = -93.5
    p3 = 0.118

    lat_len = m1 + (m2 * cos(2 * lat)) + (m3 * cos(4 * lat)) + (
        m4 * cos(6 * lat))
    lon_len = (p1 * cos(lat)) + (p2 * cos(3 * lat)) + (p3 * cos(5 * lat))
    return lat_len, lon_len
