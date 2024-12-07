import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

from src.exception import CustomException
from src.logger import logging
from src.utilis import save_object
from src.utilis import evaluate  # You can use this if you modify it for text data

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'sentiment_model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_df, test_df):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, X_test = train_df['tweet'], test_df['tweet']  # Using tweet text as features
            y_train, y_test = train_df['sentiment'], test_df['sentiment']  # Assuming 'sentiment' is the label

            # Define the models you want to evaluate
            models = {
                'LogisticRegression': LogisticRegression(),
                'NaiveBayes': MultinomialNB(),
                'SVM': SVC()
            }

            # TF-IDF Vectorization to convert text data into numerical features
            text_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))),
                ('classifier', LogisticRegression())  # Default classifier
            ])

            # Model report for evaluation
            model_report = {}

            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")
                
                # Create a pipeline for TF-IDF and model
                text_pipeline.set_params(classifier=model)
                
                # Train the model
                text_pipeline.fit(X_train, y_train)
                
                # Predict on the test data
                y_pred = text_pipeline.predict(X_test)
                
                # Evaluate the model
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred)
                
                model_report[model_name] = accuracy
                logging.info(f'Model: {model_name}, Accuracy: {accuracy}')
                logging.info(f'Classification Report:\n{report}')
                print(f'Model: {model_name}, Accuracy: {accuracy}')
                print(f'Classification Report:\n{report}')

            # Get the best model based on accuracy
            best_model_name = max(model_report, key=model_report.get)
            best_model_accuracy = model_report[best_model_name]

            logging.info(f'Best Model Found: {best_model_name} with Accuracy: {best_model_accuracy}')

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=text_pipeline
            )

            logging.info(f'Saved the best model to {self.model_trainer_config.trained_model_file_path}')

        except Exception as e:
            logging.error('Exception occurred during model training')
            raise CustomException(e, sys)
