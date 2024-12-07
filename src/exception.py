import sys
from src.logger import logging


def error_message_detail(error, error_detail: sys):
    """
    Generates a detailed error message with information about
    the file name, line number, and error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = (
        f"Error occurred in python script: [{file_name}] "
        f"at line number: [{exc_tb.tb_lineno}] "
        f"with error message: [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    """
    Custom exception class that captures detailed error information.
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    logging.info("Logging for NLP task has started")

    try:
        # Simulating an error in an NLP pipeline
        dataset_path = "data/tweets.csv"

        # Trying to open a file that may not exist
        try:
            with open(dataset_path, "r") as f:
                data = f.read()
        except FileNotFoundError as fe:
            raise CustomException(f"File not found: {dataset_path}", sys)

        # Simulating an invalid operation in text processing
        text = None  # Placeholder for a missing or null value
        if text is None:
            raise CustomException("Text data is missing or null", sys)

        # Division by zero example for demonstration
        result = 1 / 0  # This will raise a ZeroDivisionError

    except FileNotFoundError as e:
        logging.error("File not found error occurred")
        raise CustomException(e, sys)

    except ZeroDivisionError as e:
        logging.error("Attempted to divide by zero")
        raise CustomException(e, sys)

    except Exception as e:
        logging.error("An unexpected error occurred")
        raise CustomException(e, sys)
