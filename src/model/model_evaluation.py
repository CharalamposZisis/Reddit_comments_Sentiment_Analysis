#Import necessary libraries
import os
import pickle
import logging
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from mlflow.models import infer_signature


logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model__evaluation_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        logger.debug(f"Data loaded from {file_path} with shape {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    

def load_model(model_path: str):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    
    
def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    try:
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        logger.debug(f"Vectorizer loaded from {vectorizer_path}")
        return vectorizer
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    
    
def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"Parameters retrieved from {params_path}")
        return params
    except Exception as e:
        logger.error(f"Unexpected error loaded parameters from %s", params_path)
        raise 


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.debug(f"Model evaluation completed ")
        
        return report, cm
        
    except Exception as e:
        logger.error(f"Unexpected error during model evaluation %s: {e}")
        raise
    
    
def log_confusion_matrix(cm, dataset_name):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_file_path = f'confusion_matrix_{dataset_name}.png'
        plt.savefig(cm_file_path)
        mlflow.log_artifact(cm_file_path)
        plt.close()
    
    
def save_model_info(run_id:str, model_path:str, file_path:str)-> None:
    try:
        model_info = {
            "run_id": run_id,
            "model_path": model_path
        }
        with open(file_path, 'w') as f:
            json.dump(model_info, f)
    except Exception as e:
        logger.error(f"Unexpected error while saving model info to %s: %s", file_path, e)
        raise    
        
def main():
    mlflow.set_tracking_uri("http://ec2-51-21-221-187.eu-north-1.compute.amazonaws.com:5000/")
    mlflow.set_experiment("dvc_pipeline_run")
    
    with mlflow.start_run() as run:
        try:
            params = load_params(params_path=r"D:\SQL\Sentiment-Reddit\params.yaml")
            model_params = params['model_builder']

            # Keep track of model parameters
            for key, value in model_params.items():
                mlflow.log_param(key, value)


            test_data = load_data(file_path=r"D:\SQL\Sentiment-Reddit\data\interim\test_preprocessed.csv")
            model = load_model(model_path=r"D:\SQL\Sentiment-Reddit\lgbm_model.pkl")
            vectorizer = load_vectorizer(vectorizer_path=r"D:\SQL\Sentiment-Reddit\tfidf_vectorizer.pkl")
            
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values
            
            #Create an example input for signature
            example_input = pd.DataFrame(X_test_tfidf[:5].toarray(),columns=vectorizer.get_feature_names_out())

            signature = infer_signature(example_input, model.predict(X_test_tfidf[:5]))
            
            #Log model with signature
            mlflow.sklearn.log_model(model, "model", signature=signature, input_example=example_input)
            
            #Save model info
            model_path = "lgbm_model"
            save_model_info(run.info.run_id, model_path, r"experiment_info.json") # run.info.run_id is the current run ID. These id s are unique for each run.

            #Log the vectorizer as an artifact
            mlflow.log_artifact(r"D:\SQL\Sentiment-Reddit\tfidf_vectorizer.pkl")
            report, cm = evaluate_model(model, X_test_tfidf, y_test)

            for label, metrics in report.items():
                if isinstance(metrics, dict):
                        mlflow.log_metrics({
                            f"{label}_precision": metrics['precision'],
                            f"{label}_recall": metrics['recall'],
                            f"{label}_f1-score": metrics['f1-score']
                        })
                        
            log_confusion_matrix(cm, "test")

            mlflow.set_tag("model_type","LightGBM")
            mlflow.set_tag("task","Sentiment Analysis")
            mlflow.set_tag("dataset","Youtube comments")
            
        except Exception as e:
            logger.error(f"Error in main model evaluation workflow: {e}")
            raise
        
        
if __name__ == "__main__":
    main()