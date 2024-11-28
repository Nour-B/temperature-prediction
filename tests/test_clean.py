import os
import pandas as pd
import pytest

from app.src.clean import Cleaner


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "date": [19790101, 19790102, 19790103, 19790104, 19790104],
            "cloud_cover": [2.0, 6.0, 5.0, 8.0, 7.0],
            "sunshine": [7.0, 1.7, 0.0, 0.0, 0.0],
            "global_radiation": [52.0, 27.0, 13.0, 13.0, 27.0],
            "max_temp": [2.3, 1.6, 1.3, -0.3, 2.4],
            "mean_temp": [-4.1, -2.6, -2.8, -2.6, None],
            "min_temp": [-7.5, -7.5, -7.2, -6.5, -7.8],
            "precipitation": [0.4, 0.0, 0.0, 0.0, 0.3],
            "pressure": [101900.0, 102530.0, 102050.0, 100840.0, 100840.0],
            "snow_depth": [9.0, 8.0, 4.0, 2.0, 3.0],
        }
    )


@pytest.fixture()
def cleaner(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "csv_test_file.txt"
    return Cleaner(p)


def test_clean_data(cleaner, sample_data):
    cleaned_data = cleaner.clean_data(sample_data.copy())

    # Check if the columns month and year are created
    assert "month" in cleaned_data.columns
    assert "year" in cleaned_data.columns

    # Check if date is converted to datetime
    assert cleaned_data["date"].dtype == "datetime64[ns]"

    # Check if null values for mean_temp column are deleted
    assert not cleaned_data["mean_temp"].isnull().any()
    assert cleaned_data.shape == (4, 12)

    # Check if the CSV file exists
    assert os.path.exists(cleaner.path)

    # Check the content of the saved CSV file
    df = pd.read_csv(cleaner.path)
    assert list(df.columns) == [
        "date",
        "cloud_cover",
        "sunshine",
        "global_radiation",
        "max_temp",
        "mean_temp",
        "min_temp",
        "precipitation",
        "pressure",
        "snow_depth",
        "month",
        "year",
    ]
    assert df.shape == (4, 12)
    assert df.iloc[0].tolist() == ["1979-01-01", 2.0, 7.0, 52.0, 2.3, -4.1, -7.5, 0.4, 101900.0, 9.0, 1, 1979]
