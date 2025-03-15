from tensorflow.keras.models import load_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    model = load_model('D:/Project/Mini Project 2/Modal/breast_cancer.keras', compile=False)
    logger.info("Model loaded successfully")
    print(model.summary())
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")