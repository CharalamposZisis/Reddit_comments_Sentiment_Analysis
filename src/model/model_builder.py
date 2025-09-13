import numpy as np
import pandas as pd
import os 
import logging
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier


logger = logging.getLogger("model_builder")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path:str) -> dict:
    try:
        import yaml
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

def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple):
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        
        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values
        
        X_train_tfidf = vectorizer.fit_transform(X_train)

        logger.debug(f"TF-IDF transformation complete. Train shape: {X_train_tfidf.shape} fitted with max_features={max_features} and ngram_range={ngram_range}")

        with open(os.path.join(get_root_directory(), r"D:\SQL\Sentiment-Reddit\tfidf_vectorizer.pkl"), 'wb') as f:
            pickle.dump(vectorizer, f)
        
        logger.debug("TF-IDF vectorization completed.")
        
        return X_train_tfidf, y_train
    except Exception as e:
        logger.error(f"Error in TF-IDF vectorization: {e}")
        raise

def train_lgbm(X_train:np.array, y_train:np.array, learning_rate: float, max_depth: int,n_estimators: int) -> LGBMClassifier:
    
    try:
        best_model = LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric='multi_logloss',
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            reg_alpha=0.1,   # L1 penalty on leaf weights
            reg_lambda=0.1,  # L2 penalty on leaf weights
            is_unbalance=True, # handle unbalanced classes
        )
        best_model.fit(X_train, y_train)
        logger.debug("LightGBM model training completed.")
        return best_model
    except Exception as e:
        logger.error(f"Error training LightGBM model: {e}")
        raise

def save_model(model, file_path: str):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug(f"Model saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise
    
def get_root_directory() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir,'../../..'))


def main():
    try:
        root_dir = get_root_directory()

        params = load_params(os.path.join(root_dir, r"D:\SQL\Sentiment-Reddit\params.yaml"))
        max_features = params['model_builder']['max_features']
        ngram_range = tuple(params['model_builder']['ngram_range'])
        
        learning_rate = params['model_builder']['learning_rate']
        max_depth = params['model_builder']['max_depth']
        n_estimators = params['model_builder']['n_estimators']

        train_data = pd.read_csv(os.path.join(root_dir, r"D:\SQL\Sentiment-Reddit\data\interim\train_preprocessed.csv"))

        X_train_tfidf, y_train = apply_tfidf(train_data, max_features, ngram_range)

        best_model = train_lgbm(X_train_tfidf, y_train, learning_rate, max_depth, n_estimators)

        save_model(best_model, os.path.join(root_dir, r'D:\SQL\Sentiment-Reddit\lgbm_model.pkl'))

    except Exception as e:
        logger.error(f"Error in main model building workflow: {e}")
        raise
    
if __name__ == "__main__":
    main()