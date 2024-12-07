import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utilis import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, text_features):
        try:
            # Load preprocessor and model (for NLP tasks, we usually have a tokenizer and model)
            preprocessor_path = os.path.join('artifacts', 'text_preprocessor.pkl')  # Text vectorizer or tokenizer
            model_path = os.path.join('artifacts', 'sentiment_model.pkl')  # Trained sentiment analysis model

            preprocessor = load_object(preprocessor_path)  # This could be a TfidfVectorizer or tokenizer
            model = load_object(model_path)  # Trained sentiment analysis model

            # Preprocess the text features (e.g., tokenize, vectorize, etc.)
            processed_text = preprocessor.transform(text_features)

            # Predict sentiment (0: Negative, 1: Positive, 2: Neutral, etc.)
            prediction = model.predict(processed_text)
            
            # Mapping output labels (depending on model's output, you can map them to actual sentiments)
            sentiment_mapping = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
            sentiment = sentiment_mapping.get(prediction[0], "Unknown")

            return sentiment

        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, tweet: str):
        self.tweet = tweet

    def get_data_as_dataframe(self):
        try:
            # Create a dictionary for input features, in this case, only 'tweet'
            custom_data_input_dict = {'tweet': [self.tweet]}
            
            # Convert it into a pandas dataframe
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occurred in generating dataframe for prediction')
            raise CustomException(e, sys)

