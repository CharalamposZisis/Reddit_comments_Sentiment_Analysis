import json
import logging
import os
import mlflow


mlflow.set_tracking_uri("http://ec2-51-21-221-187.eu-north-1.compute.amazonaws.com:5000/")

logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model__registration_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as f:
            model_info = json.load(f)
        logger.debug(f"Model info loaded from {file_path}")
        return model_info
    except FileNotFoundError:
        logger.error(f"File {file_path} not found.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    
def register_model(model_name: str, model_info: dict):
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}" # uri unique resource identifier
        model_version = mlflow.register_model(model_uri, model_name) 
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logger.debug(f"Model {model_name} version {model_version.version} registered and transitioned to Production stage.")
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise
    
    
def main():
    try:
        model_info = load_model_info(r"experiment_info.json")
        model_name = "yt_chrome_plugin_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error in main execution: {e}")
        
        
if __name__ == "__main__":
    main()