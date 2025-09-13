import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
import os

# Configure logging
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)  

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path='params.yaml'):
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"Parameters retrieved from {params_path}")
        return params
    except FileNotFoundError:
        logger.error(f"Configuration file {params_path} not found.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    
def load_data(data_url: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(data_url)
        logger.debug(f"Data loaded from {data_url} with shape {data.shape}")
        return data
    except FileNotFoundError:
        logger.error(f"Data file {data_url} not found.")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"No data: {data_url} is empty.")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = df[df['clean_comment'].str.strip() != '']
        logger.debug("Data preprocessing completed: missing values dropped, duplicates removed, empty strings filtered")
        return df
    except KeyError as e:
        logger.error(f"Key error: {e}")   
        raise
    except Exception as e:
        logger.error(f"Unexpected error during preprocessing: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str):
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_path = os.path.join(raw_data_path, 'train.csv')
        test_path = os.path.join(raw_data_path, 'test.csv')

        train_data.to_csv(train_path, index=False)   
        test_data.to_csv(test_path, index=False)    

        logger.debug(f"Training data saved to {train_path} with shape {train_data.shape}")
        logger.debug(f"Testing data saved to {test_path} with shape {test_data.shape}")
    except Exception as e:
        logger.error(f"Unexpected error while saving data: {e}")
        raise

def main():
    try:
        params = load_params(params_path=r"D:\SQL\Sentiment-Reddit\params.yaml")
        test_size = params['data_ingestion']['test_size']

        df = load_data(data_url="https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv")  # ✅ fixed

        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        logger.debug(f"Data split into train and test sets with test size {test_size}")

        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')
        save_data(train_data, test_data, data_path=data_path)
        
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {e}")
        print(f'An error occurred: {e}')

if __name__ == "__main__":
    main()


