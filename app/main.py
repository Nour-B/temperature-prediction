import logging

from src.clean import Cleaner
from src.load import Loader
from src.predict import Predictor
from src.train import Trainer


logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')


def main():
        
        # Load the raw data
        data = Loader()
        raw_data = data.load_raw_data()
        logging.info("Data loading completed successfully!")

        # Clean data
        cleaner = Cleaner()
        cleaner.clean_data(raw_data)
        logging.info("Data cleaning completed successfully!")

        # train the models  
        trainer = Trainer()
        clean_data = data.load_clean_data()
        X_train, X_test, y_train, y_test = trainer.prepare_data(clean_data)
        logging.info("Preparing data completed successfully!")
        trainer.train_save_models(X_train, y_train)
        logging.info("Model training and saving completed successfully!")
        
        # Evaluate the model
        predictor = Predictor()
        predictor.evaluate_model(X_test, y_test)
        logging.info("Model evaluation completed successfully")

        
if __name__ == "__main__":
    main()
   
