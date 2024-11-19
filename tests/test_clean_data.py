import pandas as pd
import pytest

from app.src.clean import Cleaner
from app.src.train import Trainer


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "date": [19790101, 19790102, 19790103, 19790104],
            "cloud_cover": [2.0, 6.0, 5.0, 8.0],
            "sunshine": [7.0, 1.7, 0.0, 0.0],
            "global_radiation": [52.0, 27.0, 13.0, 13.0],
            "max_temp": [2.3, 1.6, 1.3, -0.3],
            "mean_temp": [-4.1, -2.6, -2.8, -2.6],
            "min_temp": [-7.5, -7.5, -7.2, -6.5],
            "precipitation": [0.4, 0.0, 0.0, 0.0],
            "pressure": [101900.0, 102530.0, 102050.0, 100840.0],
            "snow_depth": [9.0, 8.0, 4.0, 2.0],
        }
    )


@pytest.fixture
def cleaner():
    return Cleaner()


@pytest.fixture
def trainer():
    return Trainer()


def test_clean_data(cleaner, sample_data):
    cleaned_data = cleaner.clean_data(sample_data.copy())

    # Check if the columns month and year are created
    assert "month" in cleaned_data.columns
    assert "year" in cleaned_data.columns

    # Check if date is converted to datetime
    assert cleaned_data["date"].dtype == "datetime64[ns]"

    # Check if null values for mean_temp column are deleted
    assert not cleaned_data["mean_temp"].isnull().any()


""" def test_prepare_data(trainer,cleaner, sample_data):
    cleaned_data = cleaner.clean_data(sample_data.copy())
    X_train, X_test, _, _ = trainer.prepare_data(cleaned_data)

    # Check if the columns are dropped
    assert 'snow_depth' not in X_train.columns
    assert 'min_temp' not in X_train.columns
    assert 'max_temp' not in X_train.columns
    assert 'year' not in X_train.columns
    assert 'date' not in X_train.columns
"""
