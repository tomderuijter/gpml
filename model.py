"""Module for geo-spatial Gaussian Processes."""
from sklearn import gaussian_process


"""
1 - First normalize geo-spatial coordinates to [-1, +1] scale for ease of processing.

"""

def gpml():
    gp = gaussian_process.GaussianProcess(
        regr='constant',
        corr='squared_exponential',
        beta0=None,  # Regression weight of Ordinary Kriging
        storage_mode='full',  # Store Cholesky decomposition of Corr.
        theta0=5e-1,  # Parameters of autocorrelation
        # thetaL=1e-4,          # Lower-bound parameters
        # thetaU=1,             # Upper-bound parameters
        normalize=True,  # Standard Gaussian normalization of input
        nugget=1e-3,  # 2.22e-15,
        optimizer='fmin_cobyla',
        verbose=True
    )
    return gp


def to_spatial_dataset(df, request):
    assert 'longitude' in df.columns and 'latitude' in df.columns
    spatial_df = df[df.valid_date.apply(lambda d: str(d).startswith(request['valid_date']))].copy()
    assert len(spatial_df) > 0, "date '%s' not contained in dataset." % request['valid_date']

    # Turn valid date into valid hour
    spatial_df['valid_hour'] = spatial_df['valid_date'].apply(lambda s: int(str(s)[-2:]))
    spatial_df['valid_month'] = spatial_df['valid_date'].apply(lambda s: int(str(s)[4:6]))
    spatial_df = spatial_df.drop('valid_date', axis=1)
    spatial_df = spatial_df.drop('station_id', axis=1)

    # TODO TdR 17/07/16: Randomly distort latitude and longitude by a small offset (100m).

    # TODO TdR 17/07/16: preserve original row-mapping for later reference
    return spatial_df


def calculate_error(df, request):
    predictant = request['predictant']
    model = request['model_name']
    obs_col = predictant + '_observed'

    main_predictor = predictant + '_' + model

    error = df[obs_col] - df[main_predictor]
    return error
