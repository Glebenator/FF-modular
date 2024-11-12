import logging
import os
from datetime import datetime
from config.settings import PathConfig

def setup_logging():
    """Configure logging for the application."""
    os.makedirs(PathConfig.LOG_DIR, exist_ok=True)
    
    log_file = os.path.join(
        PathConfig.LOG_DIR,
        f'motion_barcode_{datetime.now().strftime("%Y%m%d")}.log'
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)