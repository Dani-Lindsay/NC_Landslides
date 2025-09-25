#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 12:45:00 2025

@author: andrewmcnab
"""

import logging
import os

def setup_logger(log_dir: str = "logs", log_name: str = "app.log") -> logging.Logger:
    """
    Configure logging to write INFO+ messages to both console and file.
    
    Args:
        log_dir: Directory where log files will be stored.
        log_name: Log file name.
    Returns:
        A configured Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger("my_app")
    logger.setLevel(logging.INFO)   # Minimum level to capture
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # File handler (rotates daily if you want rotation)
        fh = logging.FileHandler(log_path, mode="a")
        fh.setLevel(logging.INFO)

        # Formatter: timestamp, level, message
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        #logger.addHandler(ch)
        logger.addHandler(fh)

    return logger