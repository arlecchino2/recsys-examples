import os
import logging
from datetime import datetime

# Set up logging directory and file
log_dir = "logs"
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"{log_dir}/test_log_{current_time}.log"

# Create logs directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logger
logger = logging.getLogger("TestLogger")
logger.setLevel(logging.INFO)

# Check if handlers are already added. If not, add a file handler
if not logger.hasHandlers():
    try:
        file_handler = logging.FileHandler(log_file, mode='a', encoding=None, delay=False)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
    except Exception as e:
        print("Error setting up log file handler:", e)

# Log a test message
logger.info("This is a test log entry.")

print(f"Log file should be created at: {log_file}")
