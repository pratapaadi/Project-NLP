from sklearn.feature_extraction.text import TfidfVectorizer  # For text vectorization
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer  # For combining different preprocessing steps
import pandas as pd
import numpy as np
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utilis import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info('Data Transformation initiated')

            # Define text columns (assuming 'tweet' column exists in the dataset)
            text_columns = ['tweet']  # This column contains the text of the tweets
            target_column_name = 'sentiment'  # This is the target column (sentiment label)

            logging.info('Pipeline Initiated')

            # Text Pipeline (For tweet text preprocessing)
            text_pipeline = Pipeline(
                steps=[
                    ('tfidf', TfidfVectorizer(
                        stop_words='english',  # Remove common English stopwords
                        max_features=5000,  # Limit the number of features
                        ngram_range=(1, 2)  # Use unigrams and bigrams
                    ))
                ]
            )

            # Column transformer to apply text preprocessing
            preprocessor = ColumnTransformer([
                ('text_pipeline', text_pipeline, text_columns)  # Apply the text pipeline to the tweet column
            ])

            return preprocessor

        except Exception as e:
            logging.error("Error in Data Transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read the CSV files containing the tweet data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            # Get the preprocessing pipeline
            preprocessing_obj = self.get_data_transformation_obj()

            # Assuming 'sentiment' is the target and 'tweet' is the feature
            target_column_name = 'sentiment'
            drop_columns = [target_column_name, 'id']  # Drop 'id' column (if it exists)

            # Prepare features and target columns for both train and test data
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply preprocessing pipeline
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features and target labels into a single array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessing object (e.g., the vectorizer)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle created and saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("Exception occurred in initiate_data_transformation")
            raise CustomException(e, sys)
