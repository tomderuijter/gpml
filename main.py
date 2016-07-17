"""Main module."""
import logging

# User modules
import fs_loader
import model
import sklearn.metrics
import sklearn.cross_validation


def plot(df):
    station_ids = df.station_id.unique()
    # Select single lat-lon for every station id.

    # Plot stations on map.


def main(request):

    dataset = fs_loader.load_dataset(test_request)

    # Modify dataset to fit geo-spatial problem
    dataset = model.to_spatial_dataset(dataset, request)
    dataset[request['predictant'] + '_observed'] = model.calculate_error(dataset, request)

    # Prepare dataset for modelling
    # TODO TdR 17/07/16: Move
    y = dataset[[request['predictant'] + '_observed']]
    X = dataset.drop([request['predictant'] + '_observed'], axis=1)
    # Temporary for build-up
    X = X.drop(
        [
            'elevation',
            'forecast_hour',
            'TT2m_ModelMix',
            'FF10m_ModelMix',
            'valid_hour',
            'valid_month'
        ],
        axis=1
    )
    # Temporary fix for scikit-learn 0.17.1
    duplicates = X.duplicated()
    y = y.loc[~duplicates].values
    X = X.loc[~duplicates].values
    logging.info("Dataset has %d samples and %d features." % X.shape)

    # Make train / test split
    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(
        X, y, test_size=0.33)

    # Train model
    gpml = model.gpml()
    logging.info("Training GP..")
    gpml.fit(X_train, y_train)
    logging.info("Done training GP.")

    # Predict model
    preds = gpml.predict(X_train)

    # Validate model
    logging.info("MAE: %.3f.", (sklearn.metrics.mean_absolute_error(y_true=y_train, y_pred=preds)))

    """
    Overall structure to define.

    - Write abstraction layer around GPML module using sklearn.

    - Write data abstraction layer.
        - Station meta data loader
        - Dataset loader
        - Dataset filter based on region and time
        - Dataset splitter train-validate splitter based on region and time
        - Get test dataset (value downscaling)
        - Define pipeline
        - Define model tests
        - Define verification metrics
    """


if __name__ == "__main__":

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level='DEBUG'
    )

    test_request = {
        'predictant': 'TT2m',
        'model_elements': ['TT2m', 'FF10m'],
        'forecast_hours': [11, 12, 13],
        'model_name': 'ModelMix',
        'valid_date': '20151201'
    }
    main(test_request)
