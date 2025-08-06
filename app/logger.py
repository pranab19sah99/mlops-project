import logging

logging.basicConfig(
    filename='prediction_logs.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def log_request(input_data, prediction):
    logging.info(f"Input: {input_data} | Prediction: {prediction}")
