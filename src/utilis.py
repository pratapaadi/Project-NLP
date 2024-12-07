import sys
import os
import pickle
from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

def save_object(file_path, obj):
    try:
        # Ensure the directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object as a pickle file
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)

def evaluate(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            logging.info(f"Training and evaluating model: {model_name}")
            # Create a pipeline for text vectorization and classification
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))),
                ('classifier', model)
            ])
            # Train the model
            pipeline.fit(X_train, y_train)
            # Predict on test data
            y_pred = pipeline.predict(X_test)
            # Evaluate using accuracy
            accuracy = accuracy_score(y_test, y_pred)
            report[model_name] = accuracy
            logging.info(f"Model: {model_name}, Accuracy: {accuracy}")
            print(f"Model: {model_name}, Accuracy: {accuracy}")
        return report
    except Exception as e:
        logging.error('Exception occurred during model evaluation')
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        # Load the saved model from a pickle file
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.error('Exception occurred in load_object function')
        raise CustomException(e, sys)
