from unittest.mock import patch

import pandas as pd
import pytest

from omegaconf import OmegaConf
from sklearn.tree import DecisionTreeRegressor

from app.src.train import Trainer


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


@pytest.fixture
def cleaned_data():
    return pd.DataFrame(
        {
            "date": ["1979-01-01", "1979-01-02", "1979-01-03", "1979-01-04"],
            "cloud_cover": [2.0, 6.0, 5.0, 8.0],
            "sunshine": [7.0, 1.7, 0.0, 0.0],
            "global_radiation": [52.0, 27.0, 13.0, 13.0],
            "max_temp": [2.3, 1.6, 1.3, -0.3],
            "mean_temp": [-4.1, -2.6, -2.8, -2.6],
            "min_temp": [-7.5, -7.5, -7.2, -6.5],
            "precipitation": [0.4, 0.0, 0.0, 0.0],
            "pressure": [101900.0, 102530.0, 102050.0, 100840.0],
            "snow_depth": [9.0, 8.0, 4.0, 2.0],
            "month": [1, 1, 1, 1],
            "year": [1979, 1979, 1979, 1979],
        }
    )


@pytest.fixture
def trainer(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    models = OmegaConf.create(
        [
            {
                "name": "DecisionTreeRegressor",
                "params": {"random_state": 42, "max_depth": 10},
                "store_path": d,
                "store_filename": "DecisionTreeRegressor.pkl",
            }
        ]
    )
    return Trainer(models, test_size=0.33, random_state=42)


def test_prepare_data(trainer, cleaned_data):
    X, _, X_train, X_test, y_train, y_test = trainer.prepare_data(cleaned_data.copy())

    # Check if the columns are dropped
    assert "snow_depth" not in X.columns
    assert "min_temp" not in X.columns
    assert "max_temp" not in X.columns
    assert "year" not in X.columns
    assert "date" not in X.columns

    assert X_train.dtype == "float64"
    assert X_test.dtype == "float64"
    assert y_train.dtype == "float64"
    assert y_test.dtype == "float64"

    trainer.train_save_models(X_train, y_train)


@patch.object(DecisionTreeRegressor, "fit", return_value=None)
@patch("joblib.dump", return_value=[])
def test_train_save_models(dump, fit, trainer):
    trainer.train_save_models([], [])
    assert dump.called
    assert fit.called
