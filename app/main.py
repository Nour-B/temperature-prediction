import logging
import os

import functions_framework

from omegaconf import OmegaConf
from src.clean import Cleaner
from src.load import Loader
from src.predict import Predictor
from src.train import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


@functions_framework.cloud_event
def run(request):
    main()


def main():
    with open(f"{os.path.abspath(os.path.dirname(__file__))}/configs/config.yaml") as conf:
        config = OmegaConf.load(conf)

    # Load the raw data
    raw_data = Loader(config.data.raw_data_path).load_data()
    logging.info("Data loading completed successfully!")

    # Clean the raw data
    cleaner = Cleaner(config.data.clean_data_path)
    cleaned_data = cleaner.clean_data(raw_data)
    logging.info("Data cleaning completed successfully!")

    # Train the models
    trainer = Trainer(config.models, config.train.test_size, config.train.random_state)
    _, _, X_train, X_test, y_train, y_test = trainer.prepare_data(cleaned_data)
    logging.info("Data preparing completed successfully!")
    trainer.train_save_models(X_train, y_train)
    logging.info("Model training and saving completed successfully!")

    # Evaluate the models
    predictor = Predictor(config.models, config.evaluate.metric)
    predictor.evaluate_model_mlflow(X_test, y_test)
    logging.info("Model evaluation completed successfully")


if __name__ == "__main__":
    main()
