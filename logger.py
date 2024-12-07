import logging
import os
from datetime import datetime

# Set up the log file name based on the current timestamp
LOG_FILE = f"sentiment_analysis_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)  # Ensure the directory exists

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
    level=logging.INFO
)

# Add console logging for real-time feedback during pipeline execution
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s"))
logging.getLogger().addHandler(console_handler)

# Example usage for an NLP sentiment analysis pipeline
if __name__ == "__main__":
    logging.info("Sentiment Analysis pipeline started")

    try:
        logging.info("Step 1: Loading the dataset")
        dataset_path = os.path.join(os.getcwd(), "data", "tweets.csv")
        
        # Simulate dataset loading
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at path: {dataset_path}")
        
        logging.info(f"Dataset loaded successfully from {dataset_path}")

        logging.info("Step 2: Preprocessing the dataset")
        # Simulated preprocessing step
        logging.info("Removing special characters, URLs, and mentions")
        logging.info("Converting text to lowercase")
        logging.info("Dataset preprocessing completed successfully")

        logging.info("Step 3: Splitting dataset into training and testing sets")
        # Simulated train-test split
        logging.info("Train-test split completed with train size: 70% and test size: 30%")

        logging.info("Step 4: Training the sentiment analysis model")
        # Simulated model training
        logging.info("Model training completed successfully")

        logging.info("Step 5: Evaluating the model on the test set")
        # Simulated evaluation
        logging.info("Accuracy: 85%, Precision: 80%, Recall: 82%, F1-Score: 81%")

        logging.info("Step 6: Saving the trained model")
        # Simulated model saving step
        logging.info("Model saved successfully to 'artifacts/sentiment_model.pkl'")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise
