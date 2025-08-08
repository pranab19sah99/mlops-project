import logging
from logging.handlers import RotatingFileHandler

# Create logger
logger = logging.getLogger("mlops_logger")
logger.setLevel(logging.DEBUG)  # Capture all levels

# Formatter for log messages
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')

# Rotating file handler (1MB max, 5 backups)
file_handler = RotatingFileHandler(
    filename='prediction_logs.log',
    maxBytes=1_000_000,
    backupCount=5
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)  # Set level for file logging

# Console handler (for local/dev)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.DEBUG)  # More verbose in console

# Add handlers only once to prevent duplication
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# Function to log prediction requests
def log_request(input_data, prediction):
    logger.info(f"Prediction request | Input: {input_data} | Output: {prediction}")


# Optional: Function to log system-level events
def log_event(message: str, level: str = "info"):
    if level == "debug":
        logger.debug(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "critical":
        logger.critical(message)
    else:
        logger.info(message)


# Optional: Function to log exceptions with traceback
def log_exception(error_message: str):
    logger.exception(error_message)
