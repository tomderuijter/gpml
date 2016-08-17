import logging
import itertools
import numpy as np
import pandas as pd
import joblib as jl

cache_dir = './.cache/'
memory = jl.Memory(cachedir=cache_dir)


class DataDescriptor(object):
    """Simple dataset descriptor.

    This class is defined for ease of logging and use as dictionary key.
    """
    def __init__(self, element, forecast_hour):
        self.element = element
        self.forecast_hour = forecast_hour

    def __hash__(self):
        return hash((self.element, self.forecast_hour))

    def __eq__(self, other):
        return (self.element, self.forecast_hour) == \
               (other.element, other.forecast_hour)

    def __str__(self):
        return 'DataDescriptor(element: %s, forecast hour: %d)' % \
               (self. element, self.forecast_hour)


def load_station_data():
    # TODO TdR 13/07/16: fix hard-coded data path.
    station_data = pd.read_csv('data/mos_stations.csv')
    return station_data


def get_file_path(file_identifier):
    # TODO TdR 13/07/16: fix hard-coded data path.
    element = file_identifier.element
    forecast_hour = file_identifier.forecast_hour
    return 'data/mgforsys_europe/%s/fh%d_D11_B3_T2_M5.csv' % \
           (element, forecast_hour)


@memory.cache()
def load_single_file(file_identifier):
    file_path = get_file_path(file_identifier)
    logging.debug("Reading dataframe from file: %s" % file_path)
    df = pd.read_csv(
        file_path,
        na_values=['NaN'],
    )
    return df


def load_multiple_files(file_identifiers):
    datasets = {}
    success = 0
    for file_id in file_identifiers:
        try:
            file_content = load_single_file(file_id)
            datasets[file_id] = file_content
            success += 1
        except OSError:
            logging.error("could not load %s." % file_id)
    return datasets


def load_dataset_for_forecast_hour(forecast_hour, request):
    predictors = request['model_elements']
    files_to_load = get_file_identifiers(predictors, [forecast_hour])
    logging.debug("Data for forecast hour %d: loading %d file(s).." %
                  (forecast_hour, len(files_to_load)))
    dataset_map = load_multiple_files(files_to_load)
    logging.debug("Data for forecast hour %d: successfully loaded %d files." %
                  (forecast_hour, len(dataset_map)))

    data = pd.DataFrame()
    for data_id in dataset_map:
        element_data = dataset_map[data_id]
        if data.empty:
            data = element_data
        else:
            # Perform inner join on two dataframes, joining on common columns.
            before_dtypes = data.dtypes
            data = pd.merge(data, element_data, how='inner')
            assert np.all(data.dtypes[before_dtypes.index] == before_dtypes),\
                "Data types changed after merge."

    return data


def get_file_identifiers(elements, forecast_hours):
    return [
        DataDescriptor(e, h)
        for (e, h) in itertools.product(elements, forecast_hours)
    ]


def get_feature_columns(request):
    model_name = request['model_name']
    predictors = request['model_elements']
    feature_cols = [p + '_' + model_name for p in predictors]
    return feature_cols


def get_observation_column(request):
    predictant = request['predictant']
    obs_col = predictant + '_observed'
    return obs_col


def construct_dataset_column_names(request):
    observation_column = get_observation_column(request)
    feature_cols = get_feature_columns(request)
    dataset_column_names = [
        'station_id', 'latitude', 'longitude',
        'elevation', 'forecast_hour', 'valid_date', 'issue_date'
    ]
    dataset_column_names += [observation_column] + feature_cols
    return dataset_column_names


def validate_request(request):
    predictors = request['model_elements']
    predictant = request['predictant']
    assert predictant in predictors


def row_concat(dfs):
    return pd.concat(dfs, axis=0, copy=False)


def clean_data(data):
    # TODO TdR 17/07/16: impute data & outlier filtering
    # Imputing should be done per forecast hour
    nr_rows_before = len(data)
    # Drop rows having at least one NaN value.
    data = data.dropna(axis=0, how='any')
    nr_rows_after = len(data)
    logging.debug("Dropped %d rows." % (nr_rows_after - nr_rows_before))
    return data


def filter_locations(df, area):
    top_lat, bot_lat, left_lon, right_lon = area
    return df[
        (df.latitude >= bot_lat) & (df.latitude <= top_lat) &
        (df.longitude >= left_lon) & (df.longitude <= right_lon)
    ]


def load_dataset(request):
    """Load geo-spatial forecast dataset."""
    validate_request(request)

    station_data = load_station_data()
    column_names = construct_dataset_column_names(request)

    forecast_data = pd.DataFrame()
    for forecast_hour in request['forecast_hours']:
        logging.info("Loading files for forecast hour %d.." % forecast_hour)
        fh_data = load_dataset_for_forecast_hour(forecast_hour, request)

        if forecast_data.empty:
            forecast_data = fh_data
        else:
            forecast_data = row_concat([forecast_data, fh_data])

    forecast_data = pd.merge(forecast_data, station_data, how='inner')
    forecast_data = filter_locations(forecast_data, request['predict_area'])
    forecast_data = forecast_data.filter(items=column_names)
    forecast_data.valid_date = \
        pd.to_datetime(forecast_data.valid_date.astype(str), format='%Y%m%d%H')
    forecast_data = clean_data(forecast_data)

    return forecast_data


if __name__ == "__main__":
    test_request = {
        'predictant': 'TT2m',
        'model_elements': ['TT2m', 'FF10m'],
        'forecast_hours': [18],
        'model_name': 'ModelMix',
        'valid_date': '20151201',
        'predict_area': (60, 33, -12, 20),
        'predict_resolution': 0.1
    }

    df = load_dataset(test_request)
    print("Stations in set: %d" % (len(df.station_id.unique())))
