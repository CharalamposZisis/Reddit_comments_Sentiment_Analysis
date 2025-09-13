import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging
import re
import os

logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("preprocessing_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(comment):
    try:
        comment = comment.lower()
        
        comment = comment.dropna()
        
        comment = comment.strip()
        
        comment = re.sub(r'\n', ' ', comment)

        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english'))-{'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([char for char in comment.split() if char not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        
        return comment

    except Exception as e:
        logger.error(f"Error preprocessing comment: {e}")
        return comment
    
def normalize_text(df):
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_text)
        logger.debug("Text normalization completed.")
        return df
    except Exception as e:
        logger.error(f"Error normalizing text: {e}")
        raise
    
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str):
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Interim data path: {interim_data_path}")
        
        os.makedirs(interim_data_path, exist_ok=True)
        logger.debug(f"Interim directory created at {interim_data_path}")
        
        train_data.to_csv(os.path.join(interim_data_path, 'train_preprocessed.csv'), index=False)
        test_data.to_csv(os.path.join(interim_data_path, 'test_preprocessed.csv'), index=False)
        logger.debug(f"Preprocessed training data saved to {interim_data_path}")
        
    except Exception as e:
        logger.error(f"Error saving preprocessed data: {e}")
        raise
    
    
def main():
    try:
        logger.debug("Starting main preprocessing workflow.")
        train_data = pd.read_csv(r"D:\SQL\Sentiment-Reddit\data\raw\train.csv")
        test_data = pd.read_csv(r"D:\SQL\Sentiment-Reddit\data\raw\test.csv")
        logger.debug('Data loaded successfully for preprocessing.')

        train_preprocessed_data = normalize_text(train_data)
        test_preprocessed_data = normalize_text(test_data)

        save_data(train_preprocessed_data, test_preprocessed_data, data_path=r"D:\SQL\Sentiment-Reddit\data")

    except Exception as e:
        logger.error(f"Error in main preprocessing workflow: {e}")
        raise
    
    
if __name__ == "__main__":
    main()